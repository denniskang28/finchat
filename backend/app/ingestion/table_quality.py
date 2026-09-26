import base64
import io
import re

from pdfplumber.page import Page

from app.ingestion.repair_providers import (
    TableReconstructionProposal,
    TableRepairProvider,
)
from app.schemas import (
    ParsedColumn,
    ParsedRow,
    ParsedTable,
    ParsedTableArtifact,
    SourceToken,
    TableIssue,
    TableQuality,
    TableRepair,
)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_source_tokens(page: Page, table: ParsedTable) -> list[SourceToken]:
    x0, y0, x1, y1 = table.bbox
    words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
    tokens: list[SourceToken] = []
    for index, word in enumerate(words, start=1):
        center_x = (float(word["x0"]) + float(word["x1"])) / 2
        center_y = (float(word["top"]) + float(word["bottom"])) / 2
        if x0 <= center_x <= x1 and y0 <= center_y <= y1:
            tokens.append(
                SourceToken(
                    token_id=f"p{table.page_number}w{index}",
                    text=str(word["text"]),
                    bbox=(
                        float(word["x0"]),
                        float(word["top"]),
                        float(word["x1"]),
                        float(word["bottom"]),
                    ),
                )
            )
    return tokens


def detect_table_issues(table: ParsedTable) -> list[TableIssue]:
    issues: list[TableIssue] = []
    for row_position, row in enumerate(table.rows):
        first_cell = row.raw_cells[0] if row.raw_cells else None
        if not first_cell and any(value for value in row.raw_cells[1:]):
            populated_values = [value for value in row.raw_cells[1:] if value]
            looks_like_header_tier = bool(
                row_position == 0
                and len(populated_values) >= 4
                and all(re.search(r"[A-Za-z]", value) for value in populated_values)
                and max(len(value) for value in populated_values) <= 80
            )
            previous = table.rows[row_position - 1] if row_position > 0 else None
            previous_columns = {
                index for index, value in enumerate(previous.raw_cells[1:], start=1) if value
            } if previous else set()
            current_columns = {
                index for index, value in enumerate(row.raw_cells[1:], start=1) if value
            }
            is_split = bool(
                previous
                and previous.raw_cells[0]
                and previous_columns
                and current_columns
                and previous_columns.isdisjoint(current_columns)
            )
            issues.append(
                TableIssue(
                    code=(
                        "HEADER_ROW_AS_DATA"
                        if looks_like_header_tier
                        else "SPLIT_ROW" if is_split else "MISSING_ROW_LABEL"
                    ),
                    severity="HIGH",
                    message=(
                        "The first extracted data row appears to be a second header tier."
                        if looks_like_header_tier
                        else (
                            "This row appears to contain cells split from the preceding logical row."
                            if is_split
                            else "The row label is empty while other cells contain values."
                        )
                    ),
                    row_index=row.row_index,
                    column_index=0,
                    evidence={"raw_cells": row.raw_cells},
                )
            )
        for column_index, value in enumerate(row.raw_cells):
            if value and len(value) > 120:
                issues.append(
                    TableIssue(
                        code="OVERSIZED_CELL",
                        severity="HIGH",
                        message="A cell contains unusually long text and may include nearby chart or narrative content.",
                        row_index=row.row_index,
                        column_index=column_index,
                        evidence={"character_count": len(value)},
                    )
                )
    for column_index, column in enumerate(table.columns):
        if column.label.startswith("Column "):
            issues.append(
                TableIssue(
                    code="PLACEHOLDER_HEADER",
                    severity="MEDIUM",
                    message=f"Column {column_index + 1} has no extracted header.",
                    column_index=column_index,
                )
            )
        if all(
            column_index >= len(row.raw_cells) or not row.raw_cells[column_index]
            for row in table.rows
        ):
            issues.append(
                TableIssue(
                    code="EMPTY_COLUMN",
                    severity="MEDIUM",
                    message=f"Column {column_index + 1} contains no row values.",
                    column_index=column_index,
                )
            )
    return issues


def _confidence(issues: list[TableIssue]) -> float:
    penalties = {"HIGH": 0.30, "MEDIUM": 0.12, "LOW": 0.04}
    return max(0.05, 1.0 - sum(penalties[issue.severity] for issue in issues))


def _tokens_in_region(
    tokens: list[SourceToken], *, x0: float, x1: float, y0: float, y1: float
) -> list[SourceToken]:
    selected = []
    for token in tokens:
        tx0, ty0, tx1, ty1 = token.bbox
        center_x = (tx0 + tx1) / 2
        center_y = (ty0 + ty1) / 2
        if x0 <= center_x < x1 and y0 <= center_y <= y1:
            selected.append(token)
    return _reading_order(selected)


def _reading_order(tokens: list[SourceToken]) -> list[SourceToken]:
    lines: list[list[SourceToken]] = []
    for token in sorted(tokens, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        center_y = (token.bbox[1] + token.bbox[3]) / 2
        target = next(
            (
                line
                for line in lines
                if abs(
                    center_y
                    - sum((item.bbox[1] + item.bbox[3]) / 2 for item in line) / len(line)
                )
                <= 8.0
            ),
            None,
        )
        if target is None:
            lines.append([token])
        else:
            target.append(token)
    lines.sort(key=lambda line: sum((item.bbox[1] + item.bbox[3]) / 2 for item in line) / len(line))
    return [token for line in lines for token in sorted(line, key=lambda item: item.bbox[0])]


def _apply_cell_value(table: ParsedTable, row_index: int, column_index: int, value: str) -> None:
    row = next(row for row in table.rows if row.row_index == row_index)
    column = table.columns[column_index]
    row.raw_cells[column_index] = value
    row.values[column.key] = value
    row.display_values[column.key] = value
    if column_index == 0:
        row.row_label = value


def _repair_split_rows(table: ParsedTable) -> list[TableRepair]:
    repairs: list[TableRepair] = []
    merged_positions: list[int] = []
    for position in range(1, len(table.rows)):
        previous = table.rows[position - 1]
        row = table.rows[position]
        if row.raw_cells[0] or not previous.raw_cells[0]:
            continue
        previous_columns = {
            index for index, value in enumerate(previous.raw_cells[1:], start=1) if value
        }
        current_columns = {
            index for index, value in enumerate(row.raw_cells[1:], start=1) if value
        }
        if not previous_columns or not current_columns or not previous_columns.isdisjoint(current_columns):
            continue
        for column_index in sorted(current_columns):
            value = row.raw_cells[column_index]
            if value is None:
                continue
            column = table.columns[column_index]
            previous.raw_cells[column_index] = value
            previous.values[column.key] = value
            previous.display_values[column.key] = value
        repairs.append(
            TableRepair(
                operation="MERGE_SPLIT_ROW",
                source="DETERMINISTIC",
                row_index=row.row_index,
                column_index=min(current_columns),
                original_value=None,
                repaired_value=previous.row_label,
                reason="Merged complementary cells from an unlabeled continuation row into the preceding row.",
                confidence=0.99,
            )
        )
        merged_positions.append(position)
    for position in reversed(merged_positions):
        del table.rows[position]
    for index, row in enumerate(table.rows, start=1):
        row.row_index = index
    return repairs


def _repair_missing_labels(
    table: ParsedTable, tokens: list[SourceToken]
) -> list[TableRepair]:
    repairs: list[TableRepair] = []
    for row in table.rows:
        if row.raw_cells[0] or not any(row.raw_cells[1:]):
            continue
        row_bbox = row.metadata.get("source_row_bbox")
        label_x_range = row.metadata.get("label_x_range")
        if not row_bbox or not label_x_range:
            continue
        selected = _tokens_in_region(
            tokens,
            x0=float(label_x_range[0]),
            x1=float(label_x_range[1]),
            y0=float(row_bbox[1]),
            y1=float(row_bbox[3]),
        )
        if not selected:
            continue
        value = " ".join(token.text for token in selected)
        has_letters = any(re.search(r"[A-Za-z]", token.text) for token in selected)
        has_standalone_number = any(
            re.fullmatch(r"[+$-]?(?:\(?\d[\d,.]*\)?%?|n/m)", token.text, flags=re.IGNORECASE)
            for token in selected
        )
        if not has_letters or has_standalone_number or len(selected) > 18 or len(value) > 200:
            continue
        _apply_cell_value(table, row.row_index, 0, value)
        repairs.append(
            TableRepair(
                source="DETERMINISTIC",
                row_index=row.row_index,
                column_index=0,
                original_value=None,
                repaired_value=value,
                source_token_ids=[token.token_id for token in selected],
                reason="Recovered unassigned PDF words from the row's vertical band and label column.",
                confidence=0.99,
            )
        )
    return repairs


def _render_table_image(page: Page, table: ParsedTable) -> str:
    image = page.crop(table.bbox).to_image(resolution=150, antialias=True).original
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(stream.getvalue()).decode('ascii')}"


def _reconstruction_candidate(
    table: ParsedTable, issues: list[TableIssue]
) -> ParsedTable | None:
    issue_codes = {issue.code for issue in issues}
    if not {"MISSING_ROW_LABEL", "OVERSIZED_CELL"}.issubset(issue_codes):
        return None
    if len(table.columns) < 4:
        return None

    labelled_rows = [row for row in table.rows if row.raw_cells and row.raw_cells[0]]
    if len(labelled_rows) < 4:
        return None
    dense_threshold = max(4, int(len(labelled_rows) * 0.6))
    occupancies = [
        sum(
            1
            for row in labelled_rows
            if column_index < len(row.raw_cells) and row.raw_cells[column_index]
        )
        for column_index in range(len(table.columns))
    ]
    dense_width = 0
    for occupancy in occupancies:
        if occupancy < dense_threshold:
            break
        dense_width += 1
    if dense_width < 2 or dense_width >= len(table.columns):
        return None
    if any(
        occupancy > max(1, int(len(labelled_rows) * 0.2))
        for occupancy in occupancies[dense_width:]
    ):
        return None

    selected_columns = [column.model_copy(deep=True) for column in table.columns[:dense_width]]
    x_ranges = [column.metadata.get("source_x_range") for column in selected_columns]
    if any(not x_range for x_range in x_ranges):
        return None
    crop_x1 = max(float(x_range[1]) for x_range in x_ranges if x_range)
    table_width = float(table.bbox[2]) - float(table.bbox[0])
    crop_width = crop_x1 - float(table.bbox[0])
    if crop_width >= table_width * 0.95:
        return None

    rows: list[ParsedRow] = []
    for source in labelled_rows:
        cells = source.raw_cells[:dense_width]
        if len(cells) != dense_width or any(value is None for value in cells):
            return None
        values = {
            column.key: cells[index] for index, column in enumerate(selected_columns)
        }
        row = source.model_copy(deep=True)
        row.row_index = len(rows) + 1
        row.raw_cells = cells
        row.values = values
        row.display_values = values.copy()
        rows.append(row)

    candidate = table.model_copy(deep=True)
    candidate.bbox = (table.bbox[0], table.bbox[1], crop_x1, table.bbox[3])
    candidate.columns = selected_columns
    candidate.rows = rows
    return candidate


def _reconstruction_key(value: str, index: int, used: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or f"column_{index + 1}"
    if key in used:
        key = f"{key}_{index + 1}"
    used.add(key)
    return key


def _apply_llm_reconstruction(
    *,
    original: ParsedTable,
    candidate: ParsedTable,
    proposal: TableReconstructionProposal,
    provider: TableRepairProvider,
    tokens: list[SourceToken],
) -> tuple[ParsedTable, TableRepair] | None:
    width = len(candidate.columns)
    if len(proposal.columns) != width or len(proposal.rows) != len(candidate.rows):
        return None
    if any(len(row) != width for row in proposal.rows):
        return None

    token_map = {token.token_id: token for token in tokens}
    used_token_ids: set[str] = set()

    def evidence(value: object) -> tuple[str, list[SourceToken]] | None:
        if not value.source_token_ids or any(
            token_id not in token_map or token_id in used_token_ids
            for token_id in value.source_token_ids
        ):
            return None
        selected = _reading_order(
            [token_map[token_id] for token_id in value.source_token_ids]
        )
        source_text = " ".join(token.text for token in selected)
        if _normalized(source_text) != _normalized(value.value):
            return None
        x0, y0, x1, y1 = candidate.bbox
        if any(
            not (
                x0 <= (token.bbox[0] + token.bbox[2]) / 2 <= x1
                and y0 <= (token.bbox[1] + token.bbox[3]) / 2 <= y1
            )
            for token in selected
        ):
            return None
        used_token_ids.update(value.source_token_ids)
        return source_text, selected

    header_values: list[tuple[str, list[SourceToken]]] = []
    for index, proposed in enumerate(proposal.columns):
        verified = evidence(proposed)
        if verified is None:
            return None
        if not candidate.columns[index].label.startswith("Column ") and (
            _normalized(candidate.columns[index].label) != _normalized(verified[0])
        ):
            return None
        header_values.append(verified)

    verified_rows: list[list[tuple[str, list[SourceToken]]]] = []
    previous_y = float("-inf")
    for row_index, proposed_row in enumerate(proposal.rows):
        verified_row: list[tuple[str, list[SourceToken]]] = []
        x_centers: list[float] = []
        y_centers: list[float] = []
        for column_index, proposed in enumerate(proposed_row):
            verified = evidence(proposed)
            expected = candidate.rows[row_index].raw_cells[column_index]
            if verified is None or _normalized(verified[0]) != _normalized(expected or ""):
                return None
            verified_row.append(verified)
            x_centers.append(
                sum((token.bbox[0] + token.bbox[2]) / 2 for token in verified[1])
                / len(verified[1])
            )
            y_centers.extend((token.bbox[1] + token.bbox[3]) / 2 for token in verified[1])
        if any(left >= right for left, right in zip(x_centers, x_centers[1:])):
            return None
        if max(y_centers) - min(y_centers) > 20:
            return None
        row_y = sum(y_centers) / len(y_centers)
        if row_y <= previous_y:
            return None
        previous_y = row_y
        verified_rows.append(verified_row)

    used_keys: set[str] = set()
    columns: list[ParsedColumn] = []
    for index, (label, selected) in enumerate(header_values):
        source_column = candidate.columns[index]
        column = source_column.model_copy(deep=True)
        column.key = _reconstruction_key(label, index, used_keys)
        column.label = label
        column.semantic_label = label
        column.source_labels = [label]
        column.is_row_label = index == 0
        column.metadata = {
            "source_x_range": [
                min(token.bbox[0] for token in selected),
                max(token.bbox[2] for token in selected),
            ],
            "header_y_range": [
                min(token.bbox[1] for token in selected),
                max(token.bbox[3] for token in selected),
            ],
            "source_token_ids": proposal.columns[index].source_token_ids,
        }
        columns.append(column)

    rows: list[ParsedRow] = []
    for row_index, verified_row in enumerate(verified_rows, start=1):
        raw_cells = [value for value, _ in verified_row]
        values = {column.key: raw_cells[index] for index, column in enumerate(columns)}
        row_tokens = [token for _, selected in verified_row for token in selected]
        row_label = raw_cells[0]
        rows.append(
            ParsedRow(
                row_index=row_index,
                row_label=re.sub(r"\(\d+\)", "", row_label).strip(),
                values=values,
                display_values=values.copy(),
                raw_cells=raw_cells,
                is_total=row_label.lower().startswith(("total", "weighted average")),
                footnote_markers=re.findall(r"\((\d+)\)", row_label),
                metadata={
                    "source_row_bbox": [
                        min(token.bbox[0] for token in row_tokens),
                        min(token.bbox[1] for token in row_tokens),
                        max(token.bbox[2] for token in row_tokens),
                        max(token.bbox[3] for token in row_tokens),
                    ],
                    "source_token_ids": [
                        token_id
                        for cell in proposal.rows[row_index - 1]
                        for token_id in cell.source_token_ids
                    ],
                },
            )
        )

    reconstructed = candidate.model_copy(deep=True)
    reconstructed.columns = columns
    reconstructed.rows = rows
    reconstructed.extraction_method = f"{original.extraction_method}+llm.reconstructed"
    reconstructed.parse_warnings = [
        *original.parse_warnings,
        "LLM reconstructed a dense table region; every header and cell was verified against PDF tokens and geometry.",
    ]
    repair = TableRepair(
        operation="RECONSTRUCT_TABLE",
        source="LLM",
        provider=provider.name,
        model=provider.model,
        row_index=0,
        column_index=0,
        original_value=f"{len(original.columns)} columns x {len(original.rows)} rows",
        repaired_value=f"{len(columns)} columns x {len(rows)} rows",
        source_token_ids=[
            token.token_id for token in tokens if token.token_id in used_token_ids
        ],
        reason=proposal.reason or "Reconstructed the contiguous dense table region.",
        confidence=proposal.confidence,
    )
    return reconstructed, repair


def _apply_llm_repairs(
    *,
    table: ParsedTable,
    provider: TableRepairProvider,
    proposals: object,
    tokens: list[SourceToken],
) -> list[TableRepair]:
    token_map = {token.token_id: token for token in tokens}
    repairs: list[TableRepair] = []
    consumed_row_coverage: dict[int, set[int]] = {}
    columns_to_delete: set[int] = set()

    def selected_tokens(proposal: object) -> list[SourceToken]:
        selected = [
            token_map[token_id]
            for token_id in proposal.source_token_ids
            if token_id in token_map
        ]
        return _reading_order(selected)

    def supported_value(value: str | None, selected: list[SourceToken]) -> bool:
        if not value or not selected:
            return False
        evidence_text = " ".join(token.text for token in selected)
        return _normalized(evidence_text) == _normalized(value)

    def find_row(row_index: int | None):
        if row_index is None:
            return None
        return next((row for row in table.rows if row.row_index == row_index), None)

    def tokens_align_with_cell(
        row: object, column_index: int, selected: list[SourceToken]
    ) -> bool:
        row_bbox = row.metadata.get("source_row_bbox")
        column_x_range = (
            row.metadata.get("label_x_range")
            if column_index == 0
            else table.columns[column_index].metadata.get("source_x_range")
        )
        if row_bbox and any(
            not (
                float(row_bbox[1])
                <= (token.bbox[1] + token.bbox[3]) / 2
                <= float(row_bbox[3])
            )
            for token in selected
        ):
            return False
        if column_x_range and any(
            not (
                float(column_x_range[0])
                <= (token.bbox[0] + token.bbox[2]) / 2
                < float(column_x_range[1])
            )
            for token in selected
        ):
            return False
        return True

    def tokens_align_with_header(
        column_index: int,
        selected: list[SourceToken],
        consume_row_indices: list[int],
        allow_adjacent: bool,
    ) -> bool:
        column = table.columns[column_index]
        x_range = column.metadata.get("source_x_range")
        y_range = column.metadata.get("header_y_range")
        if not x_range or not y_range:
            return False
        min_x, max_x = map(float, x_range)
        if allow_adjacent:
            adjacent = [
                table.columns[index].metadata.get("source_x_range")
                for index in range(max(0, column_index - 1), min(len(table.columns), column_index + 2))
            ]
            adjacent = [value for value in adjacent if value]
            if adjacent:
                min_x = min(float(value[0]) for value in adjacent)
                max_x = max(float(value[1]) for value in adjacent)
        min_y, max_y = map(float, y_range)
        for row_index in consume_row_indices:
            row = find_row(row_index)
            row_bbox = row.metadata.get("source_row_bbox") if row else None
            if row_bbox:
                max_y = max(max_y, float(row_bbox[3]))
        return all(
            min_x <= (token.bbox[0] + token.bbox[2]) / 2 <= max_x
            and min_y <= (token.bbox[1] + token.bbox[3]) / 2 <= max_y
            for token in selected
        )

    for proposal in proposals.repairs:
        if proposal.column_index < 0 or proposal.column_index >= len(table.columns):
            continue
        operation = proposal.operation
        selected = selected_tokens(proposal)

        if operation in {"REPLACE_HEADER", "MERGE_HEADER"}:
            column = table.columns[proposal.column_index]
            if operation == "REPLACE_HEADER" and not column.label.startswith("Column "):
                continue
            if not supported_value(proposal.value, selected):
                continue
            if any(
                index != proposal.column_index
                and _normalized(other.label) == _normalized(proposal.value)
                for index, other in enumerate(table.columns)
            ):
                continue
            if not tokens_align_with_header(
                proposal.column_index,
                selected,
                proposal.consume_row_indices,
                allow_adjacent=operation == "MERGE_HEADER",
            ):
                continue
            original = column.label
            column.label = proposal.value
            column.semantic_label = proposal.value
            column.source_labels = [
                label.strip() for label in proposal.value.split("/") if label.strip()
            ]
            repairs.append(
                TableRepair(
                    operation=operation,
                    source="LLM",
                    provider=provider.name,
                    model=provider.model,
                    row_index=0,
                    column_index=proposal.column_index,
                    original_value=original,
                    repaired_value=proposal.value,
                    source_token_ids=proposal.source_token_ids,
                    reason=proposal.reason,
                    confidence=proposal.confidence,
                )
            )
            if operation == "MERGE_HEADER":
                for row_index in proposal.consume_row_indices:
                    consumed_row_coverage.setdefault(row_index, set()).add(
                        proposal.column_index
                    )
            continue

        if operation == "DELETE_EMPTY_COLUMN":
            column = table.columns[proposal.column_index]
            if not column.label.startswith("Column ") or any(
                row.raw_cells[proposal.column_index] for row in table.rows
            ):
                continue
            columns_to_delete.add(proposal.column_index)
            repairs.append(
                TableRepair(
                    operation=operation,
                    source="LLM",
                    provider=provider.name,
                    model=provider.model,
                    row_index=0,
                    column_index=proposal.column_index,
                    original_value=column.label,
                    repaired_value="Column deleted",
                    reason=proposal.reason,
                    confidence=proposal.confidence,
                )
            )
            continue

        row = find_row(proposal.row_index)
        if row is None or not supported_value(proposal.value, selected):
            continue

        original = row.raw_cells[proposal.column_index]
        if operation == "REPLACE_CELL" and original:
            continue
        if operation == "TRIM_CONTAMINATED_CELL":
            if not original or len(proposal.value) >= len(original):
                continue
        if operation == "REASSIGN_TOKEN":
            source_row = find_row(proposal.source_row_index)
            if (
                original
                or source_row is None
                or proposal.source_column_index is None
                or proposal.source_column_index < 0
                or proposal.source_column_index >= len(table.columns)
            ):
                continue
            source_value = source_row.raw_cells[proposal.source_column_index]
            if not source_value or _normalized(source_value) != _normalized(proposal.value):
                continue
            if not tokens_align_with_cell(
                source_row, proposal.source_column_index, selected
            ):
                continue
            source_column = table.columns[proposal.source_column_index]
            source_row.raw_cells[proposal.source_column_index] = None
            source_row.values[source_column.key] = None
            source_row.display_values[source_column.key] = None
        elif not tokens_align_with_cell(row, proposal.column_index, selected):
            continue
        _apply_cell_value(table, proposal.row_index, proposal.column_index, proposal.value)
        repairs.append(
            TableRepair(
                operation=operation,
                source="LLM",
                provider=provider.name,
                model=provider.model,
                row_index=proposal.row_index,
                column_index=proposal.column_index,
                original_value=original,
                repaired_value=proposal.value,
                source_token_ids=proposal.source_token_ids,
                reason=proposal.reason,
                confidence=proposal.confidence,
            )
        )

    rows_to_remove: set[int] = set()
    for row_index, covered_columns in consumed_row_coverage.items():
        row = find_row(row_index)
        if row is None:
            continue
        nonempty_columns = {
            index for index, value in enumerate(row.raw_cells) if value
        }
        if nonempty_columns and nonempty_columns.issubset(covered_columns):
            rows_to_remove.add(row_index)
    if rows_to_remove:
        table.rows = [row for row in table.rows if row.row_index not in rows_to_remove]
        for row_index, row in enumerate(table.rows, start=1):
            row.row_index = row_index

    for column_index in sorted(columns_to_delete, reverse=True):
        column = table.columns.pop(column_index)
        for row in table.rows:
            row.raw_cells.pop(column_index)
            row.values.pop(column.key, None)
            row.display_values.pop(column.key, None)
    return repairs


class TableRepairPipeline:
    def __init__(self, provider: TableRepairProvider) -> None:
        self.provider = provider

    def process(self, page: Page, raw_table: ParsedTable) -> ParsedTableArtifact:
        canonical = raw_table.model_copy(deep=True)
        tokens = extract_source_tokens(page, raw_table)
        initial_issues = detect_table_issues(raw_table)
        repairs = _repair_split_rows(canonical)
        repairs.extend(_repair_missing_labels(canonical, tokens))
        remaining = detect_table_issues(canonical)
        reconstruction_candidate = _reconstruction_candidate(canonical, remaining)
        if reconstruction_candidate is not None and self.provider.available:
            reconstruction_tokens = extract_source_tokens(page, reconstruction_candidate)
            try:
                proposal = self.provider.propose_reconstruction(
                    table=reconstruction_candidate,
                    issues=remaining,
                    tokens=reconstruction_tokens,
                    image_data_url=_render_table_image(page, reconstruction_candidate),
                )
                if proposal is not None:
                    result = _apply_llm_reconstruction(
                        original=canonical,
                        candidate=reconstruction_candidate,
                        proposal=proposal,
                        provider=self.provider,
                        tokens=reconstruction_tokens,
                    )
                    if result is not None:
                        canonical, reconstruction_repair = result
                        repairs.append(reconstruction_repair)
                        remaining = detect_table_issues(canonical)
            except Exception as exc:
                remaining.append(
                    TableIssue(
                        code="LLM_RECONSTRUCTION_FAILED",
                        severity="LOW",
                        message=(
                            f"{self.provider.name} reconstruction request failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                )
        repairable_codes = {
            "MISSING_ROW_LABEL",
            "PLACEHOLDER_HEADER",
            "EMPTY_COLUMN",
            "OVERSIZED_CELL",
            "HEADER_ROW_AS_DATA",
        }
        repairable_issues = [issue for issue in remaining if issue.code in repairable_codes]
        if repairable_issues and self.provider.available:
            try:
                proposals = self.provider.propose_repairs(
                    table=canonical,
                    issues=repairable_issues,
                    tokens=tokens,
                    image_data_url=_render_table_image(page, canonical),
                )
                repairs.extend(
                    _apply_llm_repairs(
                        table=canonical,
                        provider=self.provider,
                        proposals=proposals,
                        tokens=tokens,
                    )
                )
                remaining = detect_table_issues(canonical)
            except Exception as exc:
                remaining.append(
                    TableIssue(
                        code="LLM_REPAIR_FAILED",
                        severity="LOW",
                        message=f"{self.provider.name} repair request failed: {type(exc).__name__}",
                    )
                )
        if remaining:
            status = "NEEDS_REVIEW"
        elif repairs:
            status = "REPAIRED"
        else:
            status = "PASS"
        return ParsedTableArtifact(
            raw_table=raw_table,
            canonical_table=canonical,
            quality=TableQuality(
                status=status,
                confidence=_confidence(remaining),
                initial_issues=initial_issues,
                remaining_issues=remaining,
            ),
            repairs=repairs,
        )
