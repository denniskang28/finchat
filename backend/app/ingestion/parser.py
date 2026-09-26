import re
import uuid
from collections.abc import Iterable
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.table import Table

from app.ingestion.table_quality import TableRepairPipeline
from app.schemas import ParsedColumn, ParsedRow, ParsedTable, ParsedTableArtifact


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _slug(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "row"


def _pad(row: Iterable[object], width: int) -> list[str | None]:
    values = [_clean(value) for value in row]
    return (values + [None] * width)[:width]


class AiaPdfParser:
    def __init__(self, repair_pipeline: TableRepairPipeline | None = None) -> None:
        self.repair_pipeline = repair_pipeline

    def parse(
        self, pdf_path: Path, document_id: uuid.UUID
    ) -> tuple[list[str], list[ParsedTableArtifact]]:
        pages: list[str] = []
        parsed_tables: list[ParsedTableArtifact] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(layout=True) or ""
                pages.append(page_text.strip())
                for table in self.parse_page(page, page_number, document_id, page_text):
                    if self.repair_pipeline:
                        parsed_tables.append(self.repair_pipeline.process(page, table))
                    else:
                        parsed_tables.append(
                            ParsedTableArtifact(raw_table=table, canonical_table=table)
                        )
        return pages, parsed_tables

    def parse_page(
        self,
        page: Page,
        page_number: int,
        document_id: uuid.UUID,
        page_text: str | None = None,
    ) -> list[ParsedTable]:
        text = page_text if page_text is not None else (page.extract_text(layout=True) or "")
        if page_number == 70:
            tables = page.find_tables()
            return [self._parse_page_70(tables[0], document_id, text)] if tables else []
        if page_number == 91:
            return self._parse_page_91(page, document_id, text)
        if page_number == 93:
            chart = self._parse_page_93(page, document_id, text)
            return [chart] if chart else []

        results: list[ParsedTable] = []
        accepted_index = 0
        for candidate in page.find_tables():
            rows = candidate.extract()
            width = max((len(row) for row in rows), default=0)
            nonempty = sum(1 for row in rows for cell in row if _clean(cell))
            if len(rows) < 4 or width < 3 or nonempty < max(6, int(len(rows) * width * 0.30)):
                continue
            accepted_index += 1
            results.append(
                self._parse_generic(
                    candidate,
                    document_id,
                    page_number,
                    accepted_index,
                    text,
                )
            )
        return results

    def _parse_page_70(
        self, table: Table, document_id: uuid.UUID, page_text: str
    ) -> ParsedTable:
        extracted = table.extract()
        columns = [
            ParsedColumn(
                key="market",
                source_labels=["%"],
                label="Market",
                semantic_label="Market",
                is_row_label=True,
            ),
            self._rate_column("rdr_2010", "As at 30 Nov 2010", "Risk discount rate", "2010-11-30"),
            self._rate_column(
                "government_bond_2010",
                "As at 30 Nov 2010",
                "Long-term 10-year government bond rate",
                "2010-11-30",
            ),
            self._rate_column("risk_premium_2010", "As at 30 Nov 2010", "Risk premium", "2010-11-30"),
            self._rate_column("rdr_2025", "As at 31 Dec 2025", "Risk discount rate", "2025-12-31"),
            self._rate_column(
                "government_bond_2025",
                "As at 31 Dec 2025",
                "Long-term 10-year government bond rate",
                "2025-12-31",
            ),
            self._rate_column("risk_premium_2025", "As at 31 Dec 2025", "Risk premium", "2025-12-31"),
        ]
        rows = self._rows_from_cells(extracted[2:], columns)
        return ParsedTable(
            table_id="p70_risk_rates",
            document_id=document_id,
            page_number=70,
            bbox=tuple(float(value) for value in table.bbox),
            title="Risk Discount Rate and Risk Premium",
            context={"unit": "percent"},
            columns=columns,
            rows=rows,
            footnotes=self._note_lines(page_text),
            extraction_method="pdfplumber.lines",
        )

    @staticmethod
    def _rate_column(key: str, period: str, measure: str, as_of_date: str) -> ParsedColumn:
        return ParsedColumn(
            key=key,
            source_labels=[period, measure],
            label=f"{period} / {measure}",
            semantic_label=measure,
            unit="%",
            value_type="PERCENT",
            period_label=period,
            as_of_date=as_of_date,
        )

    def _parse_page_91(
        self, page: Page, document_id: uuid.UUID, page_text: str
    ) -> list[ParsedTable]:
        candidates = sorted(page.find_tables(), key=lambda value: value.bbox[0])
        results: list[ParsedTable] = []
        for candidate in candidates:
            is_geography = candidate.bbox[0] < 400
            dimension_key = "geography" if is_geography else "sector"
            dimension_label = "Geography" if is_geography else "Sector"
            title = "Corporate Bonds by Geography" if is_geography else "Corporate Bonds by Sector"
            table_id = (
                "p91_corporate_bonds_geography"
                if is_geography
                else "p91_corporate_bonds_sector"
            )
            extracted = [_pad(row, 3) for row in candidate.extract()]
            extracted = [row for row in extracted if any(row)]
            columns = [
                ParsedColumn(
                    key=dimension_key,
                    source_labels=[dimension_label],
                    label=dimension_label,
                    semantic_label=dimension_label,
                    is_row_label=True,
                ),
                ParsedColumn(
                    key="amount_usd_billion",
                    source_labels=["$b"],
                    label="$b",
                    semantic_label="Corporate bond amount",
                    unit="USD billion",
                    value_type="CURRENCY",
                ),
                ParsedColumn(
                    key="share_of_total",
                    source_labels=["% of total"],
                    label="% of total",
                    semantic_label="Share of total corporate bond portfolio",
                    unit="%",
                    value_type="PERCENT",
                ),
            ]
            warnings: list[str] = []
            original_width = max((len(row) for row in candidate.extract()), default=0)
            if original_width > 3:
                warnings.append(
                    "Nearby chart geometry expanded the detected table; only the first three ruled columns were retained."
                )
            results.append(
                ParsedTable(
                    table_id=table_id,
                    document_id=document_id,
                    page_number=91,
                    bbox=(
                        float(max(0, candidate.bbox[0])),
                        float(max(0, candidate.bbox[1])),
                        float(min(page.width, candidate.bbox[2])),
                        float(min(page.height, candidate.bbox[3])),
                    ),
                    title=title,
                    subtitle="Non-par and Surplus Assets",
                    context={
                        "portfolio": "Non-par and Surplus Assets",
                        "as_of_date": "2025-12-31",
                    },
                    columns=columns,
                    rows=self._rows_from_cells(extracted[1:], columns),
                    footnotes=self._note_lines(page_text),
                    extraction_method="pdfplumber.lines",
                    parse_warnings=warnings,
                )
            )
        return results

    def _parse_page_93(
        self, page: Page, document_id: uuid.UUID, page_text: str
    ) -> ParsedTable | None:
        regions = [
            ("Real Estate", (110, 100, 195, 150), []),
            ("Other", (195, 100, 250, 150), []),
            ("Equities", (90, 190, 170, 240), ["2"]),
            ("Corporate Bonds", (0, 245, 125, 305), ["1"]),
            ("Government & Government Agency Bonds", (140, 345, 255, 430), []),
        ]
        extracted_rows: list[list[str | None]] = []
        markers: dict[str, list[str]] = {}
        warnings: list[str] = []
        for label, bbox, footnote_markers in regions:
            words = page.crop(bbox).extract_words()
            percentage = next(
                (word["text"] for word in words if re.fullmatch(r"\d+(?:\.\d+)?%", word["text"])),
                None,
            )
            if percentage is None:
                warnings.append(f"Could not associate a percentage with chart label: {label}")
                continue
            extracted_rows.append([label, percentage])
            markers[label] = footnote_markers
        if not extracted_rows:
            return None
        total = sum(float(row[1].rstrip("%")) for row in extracted_rows if row[1])
        if abs(total - 100.0) > 0.01:
            warnings.append(f"Extracted allocation percentages sum to {total:g}%, not 100%.")
        columns = [
            ParsedColumn(
                key="asset_class",
                source_labels=["Asset class"],
                label="Asset class",
                semantic_label="Asset class",
                is_row_label=True,
            ),
            ParsedColumn(
                key="share_of_invested_assets",
                source_labels=["Share of invested assets"],
                label="Share of invested assets",
                semantic_label="Share of invested assets",
                unit="%",
                value_type="PERCENT",
            ),
        ]
        rows = self._rows_from_cells(extracted_rows, columns)
        for row in rows:
            row.footnote_markers = markers.get(row.row_label, [])
        return ParsedTable(
            table_id="p93_china_allocation",
            document_id=document_id,
            page_number=93,
            bbox=(0.0, 95.0, 350.0, 430.0),
            source_kind="CHART",
            title="AIA China: Prudent Investment Portfolio",
            subtitle="AIA China Invested Assets - Non-par and Surplus Assets",
            context={
                "portfolio": "AIA China invested assets, non-par and surplus assets",
                "as_of_date": "2025-12-31",
            },
            columns=columns,
            rows=rows,
            footnotes=self._note_lines(page_text),
            extraction_method="pdfplumber.positioned_text.sample_adapter",
            parse_warnings=warnings,
        )

    def _parse_generic(
        self,
        table: Table,
        document_id: uuid.UUID,
        page_number: int,
        table_index: int,
        page_text: str,
    ) -> ParsedTable:
        extracted = table.extract()
        width = max(len(row) for row in extracted)
        normalized = [_pad(row, width) for row in extracted]
        header = normalized[0]
        used_keys: set[str] = set()
        columns: list[ParsedColumn] = []
        for column_index, header_value in enumerate(header):
            label = header_value or f"Column {column_index + 1}"
            key = _slug(label)
            if key in used_keys:
                key = f"{key}_{column_index + 1}"
            used_keys.add(key)
            columns.append(
                ParsedColumn(
                    key=key,
                    source_labels=[label],
                    label=label,
                    semantic_label=label,
                    is_row_label=column_index == 0,
                )
            )
        header_cells = table.rows[0].cells if table.rows else []
        for column_index, column in enumerate(columns):
            body_cells = [
                row.cells[column_index]
                for row in table.rows[1:]
                if column_index < len(row.cells) and row.cells[column_index] is not None
            ]
            source_cells = body_cells or (
                [header_cells[column_index]]
                if column_index < len(header_cells) and header_cells[column_index] is not None
                else []
            )
            if source_cells:
                column.metadata["source_x_range"] = [
                    min(cell[0] for cell in source_cells),
                    max(cell[2] for cell in source_cells),
                ]
            if header_cells:
                visible_header_cells = [cell for cell in header_cells if cell is not None]
                if visible_header_cells:
                    column.metadata["header_y_range"] = [
                        min(cell[1] for cell in visible_header_cells),
                        max(cell[3] for cell in visible_header_cells),
                    ]
        title = self._page_title(page_text)
        context: dict[str, str] = {}
        date_match = re.search(r"(?i)as (?:of|at)\s+(31 Dec 2025|30 Nov 2010)", page_text)
        if date_match:
            context["as_of_date"] = date_match.group(1)
        parsed_rows = self._rows_from_cells(normalized[1:], columns)
        for parsed_row, source_row in zip(parsed_rows, table.rows[1:], strict=False):
            cells = [cell for cell in source_row.cells if cell is not None]
            if not cells:
                continue
            parsed_row.metadata["source_row_bbox"] = [
                min(cell[0] for cell in cells),
                min(cell[1] for cell in cells),
                max(cell[2] for cell in cells),
                max(cell[3] for cell in cells),
            ]
            parsed_row.metadata["label_x_range"] = [
                float(table.bbox[0]),
                float(cells[0][0] if source_row.cells[0] is None else source_row.cells[0][2]),
            ]
        return ParsedTable(
            table_id=f"p{page_number}_t{table_index}",
            document_id=document_id,
            page_number=page_number,
            bbox=tuple(float(value) for value in table.bbox),
            title=f"{title} - Table {table_index}",
            context=context,
            columns=columns,
            rows=parsed_rows,
            footnotes=self._note_lines(page_text),
            extraction_method="pdfplumber.lines.generic",
            parse_warnings=[
                "Generic header inference was used; verify merged headers, units, and table title in the debug UI."
            ],
        )

    @staticmethod
    def _rows_from_cells(
        cells: Iterable[Iterable[object]], columns: list[ParsedColumn]
    ) -> list[ParsedRow]:
        rows: list[ParsedRow] = []
        width = len(columns)
        for source_index, source_row in enumerate(cells, start=1):
            raw_cells = _pad(source_row, width)
            if not any(raw_cells):
                continue
            values = {column.key: raw_cells[index] for index, column in enumerate(columns)}
            row_label = raw_cells[0] or f"Row {source_index}"
            markers = re.findall(r"\((\d+)\)", row_label)
            rows.append(
                ParsedRow(
                    row_index=len(rows) + 1,
                    row_label=re.sub(r"\(\d+\)", "", row_label).strip(),
                    values=values,
                    display_values=values.copy(),
                    raw_cells=raw_cells,
                    is_total=row_label.lower().startswith(("total", "weighted average")),
                    footnote_markers=markers,
                )
            )
        return rows

    @staticmethod
    def _page_title(page_text: str) -> str:
        for line in page_text.splitlines():
            cleaned = _clean(line)
            if cleaned:
                return cleaned
        return "Untitled table"

    @staticmethod
    def _note_lines(page_text: str) -> list[str]:
        notes: list[str] = []
        capture = False
        for line in page_text.splitlines():
            cleaned = _clean(line)
            if not cleaned:
                continue
            if cleaned.startswith(("Note:", "Notes:")):
                capture = True
            if capture or re.match(r"^\(\d+\)", cleaned):
                notes.append(cleaned)
        return notes


slugify = _slug
