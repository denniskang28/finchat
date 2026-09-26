from app.ingestion.parser import slugify
from app.schemas import (
    ParsedColumn,
    ParsedRow,
    ParsedTable,
    ParsedTableArtifact,
    RowRepresentation,
    TableDebug,
)


def _comparison_key(table: ParsedTable, row: ParsedRow) -> str:
    return f"{table.table_id}:{slugify(row.row_label)}"


def render_raw_rows(
    table: ParsedTable, key_table: ParsedTable | None = None
) -> list[RowRepresentation]:
    key_rows = {row.row_index: row for row in (key_table or table).rows}
    representations: list[RowRepresentation] = []
    for row in table.rows:
        key_row = key_rows.get(row.row_index, row)
        representations.append(
            RowRepresentation(
                comparison_key=_comparison_key(key_table or table, key_row),
                row_index=row.row_index,
                row_label=row.row_label,
                content=" | ".join(value or "" for value in row.raw_cells),
            )
        )
    return representations


def render_markdown(table: ParsedTable) -> str:
    lines = [f"### {table.title}", ""]
    if table.subtitle:
        lines.extend([table.subtitle, ""])
    for key, value in table.context.items():
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    if table.context:
        lines.append("")
    lines.append("| " + " | ".join(column.label for column in table.columns) + " |")
    aligns = ["---" if column.is_row_label else "---:" for column in table.columns]
    lines.append("|" + "|".join(aligns) + "|")
    for row in table.rows:
        cells = [(value or "").replace("|", "\\|") for value in row.raw_cells]
        lines.append("| " + " | ".join(cells) + " |")
    if table.footnotes:
        lines.append("")
        lines.extend(table.footnotes)
    return "\n".join(lines)


def _render_value(column: ParsedColumn, value: str) -> str:
    if column.unit == "USD billion":
        return f"USD {value} billion"
    if column.unit == "%" and not value.endswith("%") and value.lower() != "n/a":
        return f"{value}%"
    return value


def _matching_footnotes(table: ParsedTable, row: ParsedRow) -> list[str]:
    if not row.footnote_markers:
        return []
    prefixes = tuple(f"({marker})" for marker in row.footnote_markers)
    return [note for note in table.footnotes if note.startswith(prefixes)]


def render_semantic_rows(table: ParsedTable) -> list[RowRepresentation]:
    representations: list[RowRepresentation] = []
    entity = "Chart" if table.source_kind == "CHART" else "Table"
    for row in table.rows:
        lines = [f"{entity}: {table.title}."]
        if table.subtitle:
            lines.append(f"Scope: {table.subtitle}.")
        for key, value in table.context.items():
            if key == "portfolio":
                label = "Portfolio"
            elif key == "as_of_date":
                label = "As of"
            else:
                label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}.")
        row_column = table.columns[0]
        lines.append(f"{row_column.semantic_label}: {row.row_label}.")
        for column in table.columns[1:]:
            value = row.display_values.get(column.key)
            if value is None or value == "":
                continue
            prefix = f"{column.period_label}, " if column.period_label else ""
            lines.append(
                f"{prefix}{column.semantic_label}: {_render_value(column, value)}."
            )
        for note in _matching_footnotes(table, row):
            lines.append(f"Note: {note}")
        representations.append(
            RowRepresentation(
                comparison_key=_comparison_key(table, row),
                row_index=row.row_index,
                row_label=row.row_label,
                content="\n".join(lines),
            )
        )
    return representations


def build_table_debug(table: ParsedTable | ParsedTableArtifact) -> TableDebug:
    if isinstance(table, ParsedTableArtifact):
        raw_table = table.raw_table
        canonical = table.canonical_table
        quality = table.quality
        repairs = table.repairs
    else:
        raw_table = table
        canonical = table
        quality = None
        repairs = []
    return TableDebug(
        parsed_table=canonical,
        raw_parsed_table=raw_table,
        **({"quality": quality} if quality is not None else {}),
        repairs=repairs,
        raw_rows=render_raw_rows(raw_table, canonical),
        markdown=render_markdown(canonical),
        semantic_rows=render_semantic_rows(canonical),
    )
