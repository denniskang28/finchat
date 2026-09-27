import base64
import io
import re
from collections import Counter
from threading import Lock

from pdfplumber.page import Page

from app.ingestion.repair_providers import (
    ChartExtractionProposal,
    ContentSectionProposal,
    EvidenceValue,
    PageStructureProposal,
    TableReconstructionProposal,
    TableRepairProvider,
)
from app.schemas import (
    ContentCoverage,
    ContentFact,
    ContentSection,
    PageContent,
    ParsedColumn,
    ParsedRow,
    ParsedTable,
    ParsedTableArtifact,
    SourceToken,
    TableIssue,
    TableQuality,
    TableRepair,
    UncoveredToken,
)


_PDF_RENDER_LOCK = Lock()


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
        if any(value and len(value) > 120 for value in row.raw_cells):
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
    with _PDF_RENDER_LOCK:
        image = page.crop(table.bbox).to_image(resolution=150, antialias=True).original
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(stream.getvalue()).decode('ascii')}"


def extract_page_tokens(page: Page, page_number: int) -> list[SourceToken]:
    words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
    chars = page.chars
    tokens = []
    for index, word in enumerate(words, start=1):
        sizes = [
            float(char["size"])
            for char in chars
            if float(word["x0"]) <= (float(char["x0"]) + float(char["x1"])) / 2 <= float(word["x1"])
            and float(word["top"]) <= (float(char["top"]) + float(char["bottom"])) / 2 <= float(word["bottom"])
            and char.get("size") is not None
        ]
        tokens.append(
            SourceToken(
                token_id=f"p{page_number}w{index}",
                text=str(word["text"]),
                bbox=(
                    float(word["x0"]),
                    float(word["top"]),
                    float(word["x1"]),
                    float(word["bottom"]),
                ),
                font_size=max(sizes) if sizes else None,
            )
        )
    return tokens


def _render_page_image(page: Page) -> str:
    with _PDF_RENDER_LOCK:
        image = page.to_image(resolution=150, antialias=True).original
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(stream.getvalue()).decode('ascii')}"


def _exact_text(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("\u2013", "-").replace("\u2014", "-")).casefold()


def _unit_name(value: str) -> str | None:
    compact = _exact_text(value).strip("()")
    if compact in {"$b", "us$b", "usd$b", "bn", "us$bn"}:
        return "USD billion"
    if compact in {"$m", "us$m", "usd$m", "mn", "us$mn"}:
        return "USD million"
    if compact == "%":
        return "%"
    return None


def _normalize_unit_rows(proposal: PageStructureProposal) -> PageStructureProposal:
    normalized = proposal.model_copy(deep=True)
    kept_rows: list[list[EvidenceValue]] = []
    for row in normalized.rows:
        populated = [(index, cell) for index, cell in enumerate(row) if cell.value.strip()]
        if len(populated) != 1 or _unit_name(populated[0][1].value) is None:
            kept_rows.append(row)
            continue
        unit_cell = populated[0][1]
        target = next(
            (column for column in normalized.columns if not column.is_row_label),
            normalized.columns[-1],
        )
        if target.unit is None:
            target.unit = unit_cell
        elif _exact_text(target.unit.value) != _exact_text(unit_cell.value):
            kept_rows.append(row)
    normalized.rows = kept_rows
    return normalized


def _validated_page_structure(
    *,
    proposal: PageStructureProposal,
    tokens: list[SourceToken],
    document_id: object,
    page_number: int,
    table_id: str,
    default_title: str,
    footnotes: list[str],
) -> ParsedTable:
    proposal = _normalize_unit_rows(proposal)
    width = len(proposal.columns)
    if not proposal.rows:
        raise ValueError("structure has no data rows after unit normalization")
    if any(len(row) != width for row in proposal.rows):
        raise ValueError("row width does not match the proposed columns")
    row_label_columns = [index for index, column in enumerate(proposal.columns) if column.is_row_label]
    if len(row_label_columns) > 1:
        raise ValueError("more than one row-label column was proposed")
    row_label_index = row_label_columns[0] if row_label_columns else 0

    token_map = {token.token_id: token for token in tokens}
    used_cell_token_ids: set[str] = set()
    geometry_tokens: list[SourceToken] = []

    def evidence(
        value: EvidenceValue,
        *,
        allow_empty: bool = False,
        allow_inferred: bool = False,
        allow_structural_normalization: bool = False,
        allow_unit_equivalence: bool = False,
        reserve: bool = True,
    ) -> tuple[str | None, list[SourceToken]]:
        if not value.value.strip():
            if value.source_token_ids or not allow_empty:
                raise ValueError("empty values must have no token ids")
            return None, []
        if not value.source_token_ids:
            inferred_labels = {"Row label", "Category"}
            if proposal.source_kind == "CHART":
                inferred_labels.update({"Percentage", "Share", "Share of total", "Value"})
            if allow_inferred and value.value in inferred_labels:
                return value.value, []
            raise ValueError(f"value has no source evidence: {value.value}")
        if len(set(value.source_token_ids)) != len(value.source_token_ids):
            raise ValueError(f"value repeats source token ids: {value.value}")
        if any(token_id not in token_map for token_id in value.source_token_ids):
            raise ValueError(f"value cites an unknown source token: {value.value}")
        if reserve and any(token_id in used_cell_token_ids for token_id in value.source_token_ids):
            raise ValueError(f"source token is reused across cells: {value.value}")
        selected = _reading_order([token_map[token_id] for token_id in value.source_token_ids])
        source_text = " ".join(token.text for token in selected)
        exact_match = _exact_text(source_text) == _exact_text(value.value)
        inferred_chart_label = bool(
            allow_inferred
            and proposal.source_kind == "CHART"
            and _exact_text(source_text) == f"by{_exact_text(value.value)}"
        )
        structurally_equivalent = bool(
            allow_structural_normalization
            and _normalized(source_text) == _normalized(value.value)
        )
        equivalent_unit = bool(
            allow_unit_equivalence
            and _unit_name(source_text) is not None
            and _unit_name(source_text) == _unit_name(value.value)
        )
        if not exact_match and not inferred_chart_label and not structurally_equivalent and not equivalent_unit:
            raise ValueError(f"value differs from source tokens: {value.value}")
        if reserve:
            used_cell_token_ids.update(value.source_token_ids)
            geometry_tokens.extend(selected)
        verified_value = value.value if inferred_chart_label else source_text
        return verified_value.replace("\u2013", "-").replace("\u2014", "-"), selected

    title = default_title
    if proposal.title is not None:
        title_value, _ = evidence(proposal.title, reserve=False)
        title = title_value or default_title

    used_keys: set[str] = set()
    columns: list[ParsedColumn] = []
    for index, proposed_column in enumerate(proposal.columns):
        label_proposal = proposed_column.label
        if proposal.source_kind == "CHART" and not label_proposal.source_token_ids:
            column_values = [row[index].value for row in proposal.rows if index < len(row)]
            inferred_label = (
                "Category"
                if index == row_label_index
                else "Percentage"
                if column_values and all(not value or value.endswith("%") for value in column_values)
                else "Value"
            )
            label_proposal = EvidenceValue(value=inferred_label)
        label, label_tokens = evidence(
            label_proposal,
            allow_inferred=index == row_label_index or proposal.source_kind == "CHART",
            allow_structural_normalization=True,
            reserve=False,
        )
        if label is None:
            raise ValueError("column label cannot be empty")
        unit = None
        unit_tokens: list[SourceToken] = []
        if proposed_column.unit is not None:
            inferred_percent_unit = bool(
                proposal.source_kind == "CHART"
                and proposed_column.unit.value == "%"
                and all(
                    index < len(row)
                    and (not row[index].value or row[index].value.endswith("%"))
                    for row in proposal.rows
                )
            )
            if inferred_percent_unit:
                raw_unit = "%"
            else:
                raw_unit, unit_tokens = evidence(
                    proposed_column.unit,
                    allow_unit_equivalence=True,
                    reserve=False,
                )
            unit = _unit_name(raw_unit or "")
            if unit is None:
                raise ValueError(f"unsupported or ambiguous unit: {raw_unit}")
        key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or f"column_{index + 1}"
        if key in used_keys:
            key = f"{key}_{index + 1}"
        used_keys.add(key)
        all_header_tokens = [*label_tokens, *unit_tokens]
        metadata = {"source_token_ids": [token.token_id for token in all_header_tokens]}
        if all_header_tokens:
            metadata["source_x_range"] = [
                min(token.bbox[0] for token in all_header_tokens),
                max(token.bbox[2] for token in all_header_tokens),
            ]
            metadata["header_y_range"] = [
                min(token.bbox[1] for token in all_header_tokens),
                max(token.bbox[3] for token in all_header_tokens),
            ]
        value_type = proposed_column.value_type
        if unit == "%":
            value_type = "PERCENT"
        elif unit in {"USD billion", "USD million"}:
            value_type = "CURRENCY"
        columns.append(
            ParsedColumn(
                key=key,
                source_labels=[label] if label_tokens else [],
                label=label,
                semantic_label=label,
                unit=unit,
                value_type=value_type,
                is_row_label=index == row_label_index,
                metadata=metadata,
            )
        )

    rows: list[ParsedRow] = []
    for row_index, proposed_row in enumerate(proposal.rows, start=1):
        verified = [evidence(cell, allow_empty=True) for cell in proposed_row]
        raw_cells = [value for value, _ in verified]
        row_label_value = raw_cells[row_label_index]
        if not row_label_value:
            raise ValueError(f"row {row_index} has no row label")
        row_tokens = [token for _, selected in verified for token in selected]
        values = {column.key: raw_cells[index] for index, column in enumerate(columns)}
        rows.append(
            ParsedRow(
                row_index=row_index,
                row_label=re.sub(r"\((\d+)\)", "", row_label_value).strip(),
                values=values,
                display_values=values.copy(),
                raw_cells=raw_cells,
                is_total=row_label_value.lower().startswith(("total", "weighted average")),
                footnote_markers=re.findall(r"\((\d+)\)", row_label_value),
                metadata={
                    "source_row_bbox": [
                        min(token.bbox[0] for token in row_tokens),
                        min(token.bbox[1] for token in row_tokens),
                        max(token.bbox[2] for token in row_tokens),
                        max(token.bbox[3] for token in row_tokens),
                    ],
                    "source_token_ids": [token.token_id for token in row_tokens],
                },
            )
        )

    context_values = []
    for context_value in proposal.context:
        value, _ = evidence(context_value, reserve=False)
        if value:
            context_values.append(value)
    if not geometry_tokens:
        raise ValueError("structure has no token-backed geometry")
    padding = 4.0
    bbox = (
        max(0.0, min(token.bbox[0] for token in geometry_tokens) - padding),
        max(0.0, min(token.bbox[1] for token in geometry_tokens) - padding),
        max(token.bbox[2] for token in geometry_tokens) + padding,
        max(token.bbox[3] for token in geometry_tokens) + padding,
    )
    return ParsedTable(
        table_id=table_id,
        document_id=document_id,
        page_number=page_number,
        bbox=bbox,
        source_kind=proposal.source_kind,
        title=title,
        context={"related_context": " | ".join(context_values)} if context_values else {},
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        extraction_method="llm.page_structure+pdfplumber.token_verified",
    )


def _proposal_token_ids(value: object) -> set[str]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        found = set(value.get("source_token_ids", []))
        for child in value.values():
            found.update(_proposal_token_ids(child))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for child in value:
            found.update(_proposal_token_ids(child))
        return found
    return set()


def _content_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:80] or "content"


def _validated_content_section(
    *,
    proposal: ContentSectionProposal,
    tokens: list[SourceToken],
    page_number: int,
    section_index: int,
    footnotes: list[str],
) -> ContentSection:
    token_map = {token.token_id: token for token in tokens}

    def evidence(
        value: EvidenceValue, *, allow_visual_heading: bool = False
    ) -> tuple[str, list[SourceToken]]:
        if not value.value.strip() or not value.source_token_ids:
            raise ValueError(f"content value has no source evidence: {value.value}")
        if len(set(value.source_token_ids)) != len(value.source_token_ids):
            raise ValueError(f"content value repeats source token ids: {value.value}")
        if any(token_id not in token_map for token_id in value.source_token_ids):
            raise ValueError(f"content value cites an unknown source token: {value.value}")
        selected = _reading_order([token_map[token_id] for token_id in value.source_token_ids])
        source_text = " ".join(token.text for token in selected)
        exact_match = _exact_text(source_text) == _exact_text(value.value)
        source_chars = Counter(_normalized(source_text))
        heading_chars = Counter(_normalized(value.value))
        character_overlap = (
            sum((source_chars & heading_chars).values()) / sum(heading_chars.values())
            if heading_chars
            else 0.0
        )
        visually_grounded_heading = bool(
            allow_visual_heading
            and re.search(r"[A-Za-z]", value.value)
            and not re.search(r"\d", value.value)
            and not re.search(r"\d", source_text)
            and character_overlap >= 0.7
        )
        if not exact_match and not visually_grounded_heading:
            raise ValueError(f"content value differs from source tokens: {value.value}")
        verified_text = value.value if visually_grounded_heading else source_text
        return verified_text.replace("\u2013", "-").replace("\u2014", "-"), selected

    heading_values = [
        evidence(heading, allow_visual_heading=True) for heading in proposal.heading_path
    ]
    heading_path = [value for value, _ in heading_values]
    heading = " > ".join(heading_path)
    section_id = f"p{page_number}_section{section_index}_{_content_slug(heading_path[-1])}"
    all_tokens = [token for _, selected in heading_values for token in selected]
    facts: list[ContentFact] = []
    raw_lines = [heading]
    for fact_index, proposed_fact in enumerate(proposal.facts, start=1):
        label, label_tokens = evidence(proposed_fact.label)
        value = None
        value_tokens: list[SourceToken] = []
        if proposed_fact.value is not None:
            value, value_tokens = evidence(proposed_fact.value)
        fact_tokens = [*label_tokens, *value_tokens]
        all_tokens.extend(fact_tokens)
        marker_text = f"{label} {value or ''}"
        markers = re.findall(r"\((\d+)\)", marker_text)
        matching_notes = [
            note for note in footnotes if any(note.startswith(f"({marker})") for marker in markers)
        ]
        sentence = f"Page {page_number}. Section: {heading}. {label}"
        if value:
            sentence += f": {value}"
        sentence += "."
        if matching_notes:
            sentence += " " + " ".join(f"Note: {note}" for note in matching_notes)
        fact_id = f"{section_id}_fact{fact_index}_{_content_slug(label)}"
        facts.append(
            ContentFact(
                fact_id=fact_id,
                label=label,
                value=value,
                content=sentence,
                source_token_ids=[token.token_id for token in fact_tokens],
                bbox=(
                    min(token.bbox[0] for token in fact_tokens),
                    min(token.bbox[1] for token in fact_tokens),
                    max(token.bbox[2] for token in fact_tokens),
                    max(token.bbox[3] for token in fact_tokens),
                ),
                footnotes=matching_notes,
            )
        )
        raw_lines.append(f"{label}{f' | {value}' if value else ''}")

    merged_facts: list[ContentFact] = []
    for fact in facts:
        previous = merged_facts[-1] if merged_facts else None
        if previous is not None and previous.value and fact.value is None:
            vertical_gap = fact.bbox[1] - previous.bbox[3]
            horizontal_overlap = max(
                0.0,
                min(previous.bbox[2], fact.bbox[2]) - max(previous.bbox[0], fact.bbox[0]),
            )
            narrower_width = max(
                1.0,
                min(previous.bbox[2] - previous.bbox[0], fact.bbox[2] - fact.bbox[0]),
            )
            if -4.0 <= vertical_gap <= 14.0 and horizontal_overlap / narrower_width >= 0.5:
                previous.label = f"{previous.label} - {fact.label}"
                previous.source_token_ids = [
                    *previous.source_token_ids,
                    *fact.source_token_ids,
                ]
                previous.bbox = (
                    min(previous.bbox[0], fact.bbox[0]),
                    min(previous.bbox[1], fact.bbox[1]),
                    max(previous.bbox[2], fact.bbox[2]),
                    max(previous.bbox[3], fact.bbox[3]),
                )
                previous.footnotes = list(dict.fromkeys([*previous.footnotes, *fact.footnotes]))
                continue
        merged_facts.append(fact)
    facts = merged_facts
    for fact_index, fact in enumerate(facts, start=1):
        fact.fact_id = f"{section_id}_fact{fact_index}_{_content_slug(fact.label)}"
        fact.content = f"Page {page_number}. Section: {heading}. {fact.label}"
        if fact.value:
            fact.content += f": {fact.value}"
        fact.content += "."
        if fact.footnotes:
            fact.content += " " + " ".join(f"Note: {note}" for note in fact.footnotes)
    if not all_tokens:
        raise ValueError("content section has no token-backed geometry")
    content_lines = [f"Page {page_number}. Section: {heading}."]
    content_lines.extend(
        f"{fact.label}{f': {fact.value}' if fact.value else ''}." for fact in facts
    )
    return ContentSection(
        section_id=section_id,
        page_number=page_number,
        heading_path=heading_path,
        content="\n".join(content_lines),
        raw_content="\n".join(raw_lines),
        source_token_ids=sorted({token.token_id for token in all_tokens}),
        bbox=(
            min(token.bbox[0] for token in all_tokens),
            min(token.bbox[1] for token in all_tokens),
            max(token.bbox[2] for token in all_tokens),
            max(token.bbox[3] for token in all_tokens),
        ),
        facts=facts,
    )


def _build_page_content(
    *,
    page_number: int,
    page_title: str,
    page_height: float,
    tokens: list[SourceToken],
    section_proposals: list[ContentSectionProposal],
    table_artifacts: list[ParsedTableArtifact],
    accepted_structure_token_ids: set[str],
    footnotes: list[str],
    parse_warnings: list[str],
) -> PageContent:
    sections: list[ContentSection] = []
    warnings = list(parse_warnings)
    for index, proposal in enumerate(section_proposals, start=1):
        try:
            section = _validated_content_section(
                proposal=proposal,
                tokens=tokens,
                page_number=page_number,
                section_index=index,
                footnotes=footnotes,
            )
            # Page titles and subtitles already belong to PAGE_SUMMARY. Keeping a
            # section wholly inside the title band creates duplicate retrieval hits.
            if section.bbox[3] <= page_height * 0.18:
                continue
            sections.append(section)
        except ValueError as exc:
            warnings.append(f"content section {index} rejected: {exc}")

    covered_token_ids = set(accepted_structure_token_ids)
    for section in sections:
        covered_token_ids.update(section.source_token_ids)
    for artifact in table_artifacts:
        x0, y0, x1, y1 = artifact.canonical_table.bbox
        covered_token_ids.update(
            token.token_id
            for token in tokens
            if x0 <= (token.bbox[0] + token.bbox[2]) / 2 <= x1
            and y0 <= (token.bbox[1] + token.bbox[3]) / 2 <= y1
        )
    top_context_tokens = _reading_order(
        [token for token in tokens if token.bbox[3] <= page_height * 0.18]
    )
    covered_token_ids.update(token.token_id for token in top_context_tokens)

    sizes = sorted(token.font_size for token in tokens if token.font_size is not None)
    median_size = sizes[len(sizes) // 2] if sizes else 0.0
    prominence_threshold = max(median_size * 1.35, median_size + 2.0)
    important: list[tuple[SourceToken, str]] = []
    for token in tokens:
        if token.bbox[1] >= page_height * 0.90:
            continue
        numeric = bool(re.search(r"\d", token.text)) and not re.fullmatch(r"\(\d+\)", token.text)
        prominent = bool(
            token.font_size is not None
            and token.font_size >= prominence_threshold
            and re.search(r"[A-Za-z]", token.text)
        )
        if numeric or prominent:
            important.append((token, "NUMERIC" if numeric else "PROMINENT_TEXT"))
    uncovered = [
        UncoveredToken(
            token_id=token.token_id,
            text=token.text,
            bbox=token.bbox,
            reason=reason,
        )
        for token, reason in important
        if token.token_id not in covered_token_ids
    ]
    covered_count = len(important) - len(uncovered)
    ratio = covered_count / len(important) if important else 1.0
    coverage_warnings = []
    uncovered_numeric = sum(token.reason == "NUMERIC" for token in uncovered)
    if uncovered_numeric:
        coverage_warnings.append(f"{uncovered_numeric} important numeric tokens are not assigned to a structure or content fact.")
    if ratio < 0.8:
        coverage_warnings.append(f"Important-token coverage is {ratio:.0%}, below the 80% review threshold.")

    section_summaries = []
    for section in sections:
        facts = "; ".join(
            f"{fact.label}{f': {fact.value}' if fact.value else ''}" for fact in section.facts
        )
        section_summaries.append(f"{' > '.join(section.heading_path)}: {facts}")
    table_titles = [artifact.canonical_table.title for artifact in table_artifacts]
    top_context = " ".join(token.text for token in top_context_tokens)
    summary_parts = [f"Page {page_number}: {top_context or page_title}."]
    if section_summaries:
        summary_parts.append("Sections and key facts: " + " | ".join(section_summaries) + ".")
    if table_titles:
        summary_parts.append("Tables and charts: " + " | ".join(table_titles) + ".")
    return PageContent(
        page_number=page_number,
        title=page_title,
        summary=" ".join(summary_parts),
        sections=sections,
        coverage=ContentCoverage(
            important_token_count=len(important),
            covered_important_token_count=covered_count,
            coverage_ratio=round(ratio, 4),
            uncovered_tokens=uncovered[:100],
            warnings=coverage_warnings,
        ),
        parse_warnings=warnings,
    )


def _bbox_overlap(left: ParsedTable, right: ParsedTable) -> float:
    x0 = max(left.bbox[0], right.bbox[0])
    y0 = max(left.bbox[1], right.bbox[1])
    x1 = min(left.bbox[2], right.bbox[2])
    y1 = min(left.bbox[3], right.bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (left.bbox[2] - left.bbox[0]) * (left.bbox[3] - left.bbox[1]))
    return intersection / area


def _reconstruction_candidate(
    table: ParsedTable, issues: list[TableIssue]
) -> ParsedTable | None:
    issue_codes = {issue.code for issue in issues}
    has_sparse_trailing_text = any(
        row.raw_cells
        and not row.raw_cells[0]
        and any(value and len(value) > 40 for value in row.raw_cells[2:])
        for row in table.rows
    )
    if not {"MISSING_ROW_LABEL", "SPLIT_ROW"}.intersection(issue_codes) or not (
        "OVERSIZED_CELL" in issue_codes or has_sparse_trailing_text
    ):
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
        return source_text.replace("\u2013", "-").replace("\u2014", "-"), selected

    header_values: list[tuple[str, list[SourceToken]]] = []
    for index, proposed in enumerate(proposal.columns):
        source_column = candidate.columns[index]
        is_inferred_row_label = (
            source_column.is_row_label and not source_column.source_labels
        )
        if (
            is_inferred_row_label
            and proposed.value == "Row label"
            and not proposed.source_token_ids
        ):
            verified = ("Row label", [])
        else:
            verified = evidence(proposed)
        if verified is None:
            return None
        if not is_inferred_row_label and not source_column.label.startswith("Column ") and (
            _normalized(source_column.label) != _normalized(verified[0])
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
        if "%" in label:
            column.unit = "%"
            column.value_type = "PERCENT"
        elif "$b" in label:
            column.unit = "USD billion"
            column.value_type = "CURRENCY"
        column.is_row_label = index == 0
        column.metadata = {
            "source_x_range": (
                [
                    min(token.bbox[0] for token in selected),
                    max(token.bbox[2] for token in selected),
                ]
                if selected
                else source_column.metadata.get("source_x_range")
            ),
            "header_y_range": (
                [
                    min(token.bbox[1] for token in selected),
                    max(token.bbox[3] for token in selected),
                ]
                if selected
                else source_column.metadata.get("header_y_range")
            ),
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


def _apply_llm_chart_extraction(
    *,
    candidate: ParsedTable,
    proposal: ChartExtractionProposal,
    provider: TableRepairProvider,
    tokens: list[SourceToken],
) -> tuple[ParsedTable, TableRepair] | None:
    if len(proposal.rows) != len(candidate.rows):
        return None
    token_map = {token.token_id: token for token in tokens}
    expected_shares = {
        str(row.metadata.get("share_token_id")): str(row.raw_cells[1])
        for row in candidate.rows
    }
    used_token_ids: set[str] = set()
    used_share_ids: set[str] = set()
    used_labels: set[str] = set()

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
        used_token_ids.update(value.source_token_ids)
        return source_text.replace("\u2013", "-").replace("\u2014", "-"), selected

    verified_rows: list[tuple[str, str, list[SourceToken]]] = []
    for proposed in proposal.rows:
        if len(proposed.share.source_token_ids) != 1:
            return None
        share_token_id = proposed.share.source_token_ids[0]
        if share_token_id not in expected_shares or share_token_id in used_share_ids:
            return None
        label = evidence(proposed.label)
        share = evidence(proposed.share)
        if label is None or share is None:
            return None
        if _normalized(share[0]) != _normalized(expected_shares[share_token_id]):
            return None
        if not re.search(r"[A-Za-z]", label[0]) or len(label[0]) > 120:
            return None
        normalized_label = _normalized(label[0])
        if normalized_label in used_labels:
            return None
        label_x = sum((token.bbox[0] + token.bbox[2]) / 2 for token in label[1]) / len(label[1])
        label_y = sum((token.bbox[1] + token.bbox[3]) / 2 for token in label[1]) / len(label[1])
        share_x = (share[1][0].bbox[0] + share[1][0].bbox[2]) / 2
        share_y = (share[1][0].bbox[1] + share[1][0].bbox[3]) / 2
        if ((label_x - share_x) ** 2 + (label_y - share_y) ** 2) ** 0.5 > 160:
            return None
        used_share_ids.add(share_token_id)
        used_labels.add(normalized_label)
        verified_rows.append(
            (
                label[0].replace("\u2013", "-").replace("\u2014", "-"),
                share[0],
                [*label[1], *share[1]],
            )
        )
    if used_share_ids != set(expected_shares):
        return None

    columns = [column.model_copy(deep=True) for column in candidate.columns]
    rows: list[ParsedRow] = []
    for row_index, (label, share, row_tokens) in enumerate(verified_rows, start=1):
        values = {columns[0].key: label, columns[1].key: share}
        rows.append(
            ParsedRow(
                row_index=row_index,
                row_label=re.sub(r"\(\d+\)", "", label).strip(),
                values=values,
                display_values=values.copy(),
                raw_cells=[label, share],
                footnote_markers=re.findall(r"\((\d+)\)", label),
                metadata={
                    "source_row_bbox": [
                        min(token.bbox[0] for token in row_tokens),
                        min(token.bbox[1] for token in row_tokens),
                        max(token.bbox[2] for token in row_tokens),
                        max(token.bbox[3] for token in row_tokens),
                    ],
                    "source_token_ids": [token.token_id for token in row_tokens],
                },
            )
        )
    reconstructed = candidate.model_copy(deep=True)
    reconstructed.rows = rows
    reconstructed.extraction_method = "pdfplumber.positioned_text+llm.chart_series"
    reconstructed.parse_warnings = [
        "LLM associated chart labels and shares; every value was verified against PDF tokens and geometry."
    ]
    repair = TableRepair(
        operation="RECONSTRUCT_TABLE",
        source="LLM",
        provider=provider.name,
        model=provider.model,
        row_index=0,
        column_index=0,
        original_value=f"percentage chart candidate x {len(candidate.rows)} rows",
        repaired_value=f"2 columns x {len(rows)} rows",
        source_token_ids=[
            token.token_id for token in tokens if token.token_id in used_token_ids
        ],
        reason=proposal.reason or "Associated percentage chart labels with their shares.",
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
    def __init__(self, provider: TableRepairProvider, *, llm_primary: bool = False) -> None:
        self.provider = provider
        self.llm_primary = llm_primary

    def process_page(
        self,
        *,
        page: Page,
        raw_tables: list[ParsedTable],
        document_id: object,
        page_number: int,
        page_title: str,
        footnotes: list[str],
    ) -> list[ParsedTableArtifact]:
        artifacts, _ = self.process_page_with_content(
            page=page,
            raw_tables=raw_tables,
            document_id=document_id,
            page_number=page_number,
            page_title=page_title,
            footnotes=footnotes,
        )
        return artifacts

    def process_page_with_content(
        self,
        *,
        page: Page,
        raw_tables: list[ParsedTable],
        document_id: object,
        page_number: int,
        page_title: str,
        footnotes: list[str],
    ) -> tuple[list[ParsedTableArtifact], PageContent]:
        tokens = extract_page_tokens(page, page_number)
        if not self.llm_primary or not self.provider.available:
            artifacts = [self.process(page, table) for table in raw_tables]
            return artifacts, _build_page_content(
                page_number=page_number,
                page_title=page_title,
                page_height=float(page.height),
                tokens=tokens,
                section_proposals=[],
                table_artifacts=artifacts,
                accepted_structure_token_ids=set(),
                footnotes=footnotes,
                parse_warnings=["Content-section extraction was unavailable; only deterministic structures were used."],
            )

        failure_messages: list[str] = []
        try:
            response = self.provider.propose_page_structures(
                page_number=page_number,
                page_width=float(page.width),
                page_height=float(page.height),
                tokens=tokens,
                image_data_url=_render_page_image(page),
            )
        except Exception as exc:
            response = None
            detail = str(exc).strip()
            failure_messages.append(
                f"provider request failed: {type(exc).__name__}{f': {detail}' if detail else ''}"
            )

        validated: list[tuple[ParsedTable, float]] = []
        accepted_structure_token_ids: set[str] = set()
        if response is not None:
            kind_counts = {"TABLE": 0, "CHART": 0}
            for index, proposal in enumerate(response.structures, start=1):
                kind_counts[proposal.source_kind] += 1
                suffix = "t" if proposal.source_kind == "TABLE" else "chart"
                table_id = f"p{page_number}_{suffix}{kind_counts[proposal.source_kind]}"
                try:
                    table = _validated_page_structure(
                        proposal=proposal,
                        tokens=tokens,
                        document_id=document_id,
                        page_number=page_number,
                        table_id=table_id,
                        default_title=page_title,
                        footnotes=footnotes,
                    )
                    validated.append((table, proposal.confidence))
                    accepted_structure_token_ids.update(_proposal_token_ids(proposal))
                except ValueError as exc:
                    failure_messages.append(f"structure {index} rejected: {exc}")

        unmatched_raw = set(range(len(raw_tables)))
        artifacts: list[ParsedTableArtifact] = []
        for canonical, confidence in validated:
            matches = sorted(
                (
                    (_bbox_overlap(canonical, raw_tables[index]), index)
                    for index in unmatched_raw
                ),
                reverse=True,
            )
            match_index = matches[0][1] if matches and matches[0][0] >= 0.25 else None
            if match_index is None:
                raw = canonical.model_copy(deep=True)
            else:
                unmatched_raw.remove(match_index)
                raw = raw_tables[match_index]
            artifacts.append(
                ParsedTableArtifact(
                    raw_table=raw,
                    canonical_table=canonical,
                    quality=TableQuality(
                        status="PASS",
                        confidence=confidence,
                        initial_issues=detect_table_issues(raw) if match_index is not None else [],
                        remaining_issues=[],
                    ),
                )
            )

        used_table_ids = {
            artifact.canonical_table.table_id for artifact in artifacts
        }
        fallback_reason = "; ".join(failure_messages[:3])
        if response is not None and not response.structures:
            fallback_reason = "provider returned no page structures"
        for index in sorted(unmatched_raw):
            artifact = self.process(page, raw_tables[index])
            if artifact.canonical_table.table_id in used_table_ids:
                suffix = "chart" if artifact.canonical_table.source_kind == "CHART" else "t"
                counter = 1
                while f"p{page_number}_{suffix}{counter}" in used_table_ids:
                    counter += 1
                unique_id = f"p{page_number}_{suffix}{counter}"
                artifact.raw_table.table_id = unique_id
                artifact.canonical_table.table_id = unique_id
            used_table_ids.add(artifact.canonical_table.table_id)
            if fallback_reason:
                artifact.canonical_table.parse_warnings.append(
                    f"LLM-primary fallback: {fallback_reason}."
                )
            artifacts.append(artifact)
        sorted_artifacts = sorted(
            artifacts,
            key=lambda artifact: (
                artifact.canonical_table.bbox[1],
                artifact.canonical_table.bbox[0],
            ),
        )
        content = _build_page_content(
            page_number=page_number,
            page_title=page_title,
            page_height=float(page.height),
            tokens=tokens,
            section_proposals=response.content_sections if response is not None else [],
            table_artifacts=sorted_artifacts,
            accepted_structure_token_ids=accepted_structure_token_ids,
            footnotes=footnotes,
            parse_warnings=failure_messages,
        )
        return sorted_artifacts, content

    def process(self, page: Page, raw_table: ParsedTable) -> ParsedTableArtifact:
        canonical = raw_table.model_copy(deep=True)
        tokens = extract_source_tokens(page, raw_table)
        initial_issues = detect_table_issues(raw_table)
        repairs: list[TableRepair] = []
        extraction_failures: list[TableIssue] = []
        is_chart_candidate = (
            raw_table.extraction_method
            == "pdfplumber.positioned_text.percentage_chart_candidate"
        )
        if is_chart_candidate and self.provider.available:
            try:
                proposal = self.provider.propose_chart_extraction(
                    table=raw_table,
                    issues=initial_issues,
                    tokens=tokens,
                    image_data_url=_render_table_image(page, raw_table),
                )
                if proposal is not None:
                    result = _apply_llm_chart_extraction(
                        candidate=raw_table,
                        proposal=proposal,
                        provider=self.provider,
                        tokens=tokens,
                    )
                    if result is not None:
                        canonical, chart_repair = result
                        repairs.append(chart_repair)
            except Exception as exc:
                extraction_failures.append(
                    TableIssue(
                        code="LLM_CHART_EXTRACTION_FAILED",
                        severity="LOW",
                        message=(
                            f"{self.provider.name} chart extraction request failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                )
        reconstruction_candidate = _reconstruction_candidate(canonical, initial_issues)
        if reconstruction_candidate is not None and self.provider.available:
            reconstruction_tokens = extract_source_tokens(page, reconstruction_candidate)
            try:
                proposal = self.provider.propose_reconstruction(
                    table=reconstruction_candidate,
                    issues=initial_issues,
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
            except Exception as exc:
                extraction_failures.append(
                    TableIssue(
                        code="LLM_RECONSTRUCTION_FAILED",
                        severity="LOW",
                        message=(
                            f"{self.provider.name} reconstruction request failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                )
        repairs.extend(_repair_split_rows(canonical))
        repairs.extend(_repair_missing_labels(canonical, tokens))
        remaining = [*detect_table_issues(canonical), *extraction_failures]
        repairable_codes = {
            "MISSING_ROW_LABEL",
            "PLACEHOLDER_HEADER",
            "EMPTY_COLUMN",
            "OVERSIZED_CELL",
            "HEADER_ROW_AS_DATA",
        }
        repairable_issues = [issue for issue in remaining if issue.code in repairable_codes]
        if repairable_issues and self.provider.available and not is_chart_candidate:
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
