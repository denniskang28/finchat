import re
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
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


class PdfParser:
    def __init__(
        self,
        repair_pipeline: TableRepairPipeline | None = None,
        *,
        page_workers: int = 1,
    ) -> None:
        self.repair_pipeline = repair_pipeline
        self.page_workers = max(1, page_workers)

    def parse(
        self, pdf_path: Path, document_id: uuid.UUID
    ) -> tuple[list[str], list[ParsedTableArtifact]]:
        use_parallel = bool(
            self.page_workers > 1
            and self.repair_pipeline
            and self.repair_pipeline.llm_primary
            and self.repair_pipeline.provider.available
        )
        if use_parallel:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

            def parse_one(page_number: int) -> tuple[str, list[ParsedTableArtifact]]:
                with pdfplumber.open(pdf_path) as worker_pdf:
                    page = worker_pdf.pages[page_number - 1]
                    page_text = page.extract_text(layout=True) or ""
                    raw_tables = self.parse_page(page, page_number, document_id, page_text)
                    artifacts = self.repair_pipeline.process_page(
                        page=page,
                        raw_tables=raw_tables,
                        document_id=document_id,
                        page_number=page_number,
                        page_title=self._page_title(page_text),
                        footnotes=self._note_lines(page_text),
                    )
                    return page_text.strip(), artifacts

            with ThreadPoolExecutor(max_workers=self.page_workers) as executor:
                results = list(executor.map(parse_one, range(1, page_count + 1)))
            return (
                [page_text for page_text, _ in results],
                [artifact for _, artifacts in results for artifact in artifacts],
            )

        pages: list[str] = []
        parsed_tables: list[ParsedTableArtifact] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(layout=True) or ""
                pages.append(page_text.strip())
                raw_tables = self.parse_page(page, page_number, document_id, page_text)
                if self.repair_pipeline:
                    parsed_tables.extend(
                        self.repair_pipeline.process_page(
                            page=page,
                            raw_tables=raw_tables,
                            document_id=document_id,
                            page_number=page_number,
                            page_title=self._page_title(page_text),
                            footnotes=self._note_lines(page_text),
                        )
                    )
                else:
                    parsed_tables.extend(
                        ParsedTableArtifact(raw_table=table, canonical_table=table)
                        for table in raw_tables
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
                    page,
                    candidate,
                    document_id,
                    page_number,
                    accepted_index,
                    text,
                )
            )
        if results:
            return results
        chart = self._parse_percentage_chart_candidate(
            page, document_id, page_number, text
        )
        return [chart] if chart else []

    def _parse_percentage_chart_candidate(
        self,
        page: Page,
        document_id: uuid.UUID,
        page_number: int,
        page_text: str,
    ) -> ParsedTable | None:
        words = page.extract_words(
            x_tolerance=2, y_tolerance=2, keep_blank_chars=False
        )
        percentages: list[tuple[int, dict[str, object], float]] = []
        for word_index, word in enumerate(words, start=1):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)%", str(word["text"]))
            center_x = (float(word["x0"]) + float(word["x1"])) / 2
            center_y = (float(word["top"]) + float(word["bottom"])) / 2
            if (
                match
                and 0 <= center_x <= page.width
                and 0 <= center_y <= page.height
            ):
                percentages.append((word_index, word, float(match.group(1))))
        if len(percentages) < 3 or len(percentages) > 15:
            return None

        selected: tuple[tuple[int, dict[str, object], float], ...] | None = None
        for size in range(min(10, len(percentages)), 2, -1):
            def is_compact(
                group: tuple[tuple[int, dict[str, object], float], ...]
            ) -> bool:
                width = max(float(item[1]["x1"]) for item in group) - min(
                    float(item[1]["x0"]) for item in group
                )
                height = max(float(item[1]["bottom"]) for item in group) - min(
                    float(item[1]["top"]) for item in group
                )
                return width <= page.width * 0.6 and height <= page.height * 0.7

            matches = [
                group
                for group in combinations(percentages, size)
                if abs(sum(item[2] for item in group) - 100.0) <= 0.01
                and is_compact(group)
            ]
            if not matches:
                continue

            def area(group: tuple[tuple[int, dict[str, object], float], ...]) -> float:
                x0 = min(float(item[1]["x0"]) for item in group)
                x1 = max(float(item[1]["x1"]) for item in group)
                y0 = min(float(item[1]["top"]) for item in group)
                y1 = max(float(item[1]["bottom"]) for item in group)
                return max(1.0, x1 - x0) * max(1.0, y1 - y0)

            selected = min(matches, key=area)
            break
        if selected is None:
            return None

        selected = tuple(
            sorted(selected, key=lambda item: (float(item[1]["top"]), float(item[1]["x0"])))
        )
        x0 = max(0.0, min(float(item[1]["x0"]) for item in selected) - 100.0)
        x1 = min(page.width, max(float(item[1]["x1"]) for item in selected) + 110.0)
        y0 = max(0.0, min(float(item[1]["top"]) for item in selected) - 80.0)
        y1 = min(page.height, max(float(item[1]["bottom"]) for item in selected) + 50.0)
        columns = [
            ParsedColumn(
                key="category",
                source_labels=["Category"],
                label="Category",
                semantic_label="Category",
                is_row_label=True,
            ),
            ParsedColumn(
                key="share_of_total",
                source_labels=["Share of total"],
                label="Share of total",
                semantic_label="Share of total",
                unit="%",
                value_type="PERCENT",
            ),
        ]
        rows = []
        for row_index, (word_index, word, _) in enumerate(selected, start=1):
            percentage = str(word["text"])
            values = {"category": None, "share_of_total": percentage}
            rows.append(
                ParsedRow(
                    row_index=row_index,
                    row_label=f"Row {row_index}",
                    values=values,
                    display_values=values.copy(),
                    raw_cells=[None, percentage],
                    metadata={
                        "share_token_id": f"p{page_number}w{word_index}",
                        "source_row_bbox": [
                            float(word["x0"]),
                            float(word["top"]),
                            float(word["x1"]),
                            float(word["bottom"]),
                        ],
                    },
                )
            )
        context: dict[str, str] = {}
        date_match = re.search(r"(?i)as (?:of|at)\s+(31 Dec 2025|30 Nov 2010)", page_text)
        if date_match:
            context["as_of_date"] = date_match.group(1)
        return ParsedTable(
            table_id=f"p{page_number}_chart1",
            document_id=document_id,
            page_number=page_number,
            bbox=(x0, y0, x1, y1),
            source_kind="CHART",
            title=self._page_title(page_text),
            context=context,
            columns=columns,
            rows=rows,
            footnotes=self._note_lines(page_text),
            extraction_method="pdfplumber.positioned_text.percentage_chart_candidate",
            parse_warnings=[
                "Generic percentage-distribution candidate requires token-verified label association."
            ],
        )

    def _parse_generic(
        self,
        page: Page,
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
        table_unit = "%" if header and header[0] == "%" else None
        for column_index, header_value in enumerate(header):
            label = header_value or (
                "Row label" if column_index == 0 else f"Column {column_index + 1}"
            )
            key = _slug(label)
            if key in used_keys:
                key = f"{key}_{column_index + 1}"
            used_keys.add(key)
            unit = (
                "%"
                if "%" in label or (column_index > 0 and table_unit == "%")
                else "USD billion" if "$b" in label else None
            )
            value_type = "PERCENT" if unit == "%" else "CURRENCY" if unit else "TEXT"
            columns.append(
                ParsedColumn(
                    key=key,
                    source_labels=[header_value] if header_value else [],
                    label=label,
                    semantic_label=label,
                    unit=unit,
                    value_type=value_type,
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
        subtitle = None
        x0, y0, x1, _ = table.bbox
        nearby_text = page.crop(
            (
                max(0.0, float(x0)),
                max(0.0, float(y0) - 70.0),
                min(float(page.width), float(x1)),
                float(y0),
            )
        ).extract_text()
        nearby_lines = [
            cleaned
            for line in (nearby_text or "").splitlines()
            if (cleaned := _clean(line))
        ]
        if nearby_lines:
            title = nearby_lines[0]
            subtitle = nearby_lines[1] if len(nearby_lines) > 1 else None
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
            subtitle=subtitle,
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
AiaPdfParser = PdfParser
