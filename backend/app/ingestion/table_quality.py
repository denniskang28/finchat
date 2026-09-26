import base64
import io
import re

from pdfplumber.page import Page

from app.ingestion.repair_providers import TableRepairProvider
from app.schemas import (
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
                    code="SPLIT_ROW" if is_split else "MISSING_ROW_LABEL",
                    severity="HIGH",
                    message=(
                        "This row appears to contain cells split from the preceding logical row."
                        if is_split
                        else "The row label is empty while other cells contain values."
                    ),
                    row_index=row.row_index,
                    column_index=0,
                    evidence={"raw_cells": row.raw_cells},
                )
            )
        for column_index, value in enumerate(row.raw_cells):
            if value and len(value) > 300:
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


def _apply_llm_repairs(
    *,
    table: ParsedTable,
    provider: TableRepairProvider,
    proposals: object,
    tokens: list[SourceToken],
) -> list[TableRepair]:
    token_map = {token.token_id: token for token in tokens}
    repairs: list[TableRepair] = []
    for proposal in proposals.repairs:
        try:
            row = next(row for row in table.rows if row.row_index == proposal.row_index)
        except StopIteration:
            continue
        if proposal.column_index < 0 or proposal.column_index >= len(table.columns):
            continue
        original = row.raw_cells[proposal.column_index]
        if original:
            continue
        selected = [token_map[token_id] for token_id in proposal.source_token_ids if token_id in token_map]
        selected = _reading_order(selected)
        evidence_text = " ".join(token.text for token in selected)
        if not selected or _normalized(evidence_text) != _normalized(proposal.value):
            continue
        row_bbox = row.metadata.get("source_row_bbox")
        label_x_range = row.metadata.get("label_x_range") if proposal.column_index == 0 else None
        if row_bbox and any(
            not (
                float(row_bbox[1])
                <= (token.bbox[1] + token.bbox[3]) / 2
                <= float(row_bbox[3])
            )
            for token in selected
        ):
            continue
        if label_x_range and any(
            not (
                float(label_x_range[0])
                <= (token.bbox[0] + token.bbox[2]) / 2
                < float(label_x_range[1])
            )
            for token in selected
        ):
            continue
        _apply_cell_value(table, proposal.row_index, proposal.column_index, proposal.value)
        repairs.append(
            TableRepair(
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
        repairable_issues = [
            issue for issue in remaining if issue.code == "MISSING_ROW_LABEL"
        ]
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
