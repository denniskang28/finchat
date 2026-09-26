import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.renderers import build_table_debug


def _parse_page_91(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[90]
        return parser.parse_page(page, 91, uuid.uuid4())


def test_page_91_detects_two_clean_tables(sample_pdf):
    tables = _parse_page_91(sample_pdf)

    assert [table.table_id for table in tables] == [
        "p91_corporate_bonds_geography",
        "p91_corporate_bonds_sector",
    ]
    assert [len(table.columns) for table in tables] == [3, 3]
    assert [len(table.rows) for table in tables] == [4, 14]
    assert tables[0].context == {
        "portfolio": "Non-par and Surplus Assets",
        "as_of_date": "2025-12-31",
    }


def test_page_91_united_states_values_and_representations(sample_pdf):
    geography = _parse_page_91(sample_pdf)[0]
    united_states = next(row for row in geography.rows if row.row_label == "United States")

    assert united_states.raw_cells == ["United States", "6.2", "22%"]
    assert united_states.display_values["amount_usd_billion"] == "6.2"
    assert united_states.display_values["share_of_total"] == "22%"

    debug = build_table_debug(geography)
    raw = next(row for row in debug.raw_rows if row.row_label == "United States")
    semantic = next(row for row in debug.semantic_rows if row.row_label == "United States")

    assert raw.comparison_key == "p91_corporate_bonds_geography:united_states"
    assert raw.content == "United States | 6.2 | 22%"
    assert "| United States | 6.2 | 22% |" in debug.markdown
    assert "Corporate bond amount: USD 6.2 billion." in semantic.content
    assert "Share of total corporate bond portfolio: 22%." in semantic.content
    assert "As of: 2025-12-31." in semantic.content


def test_page_91_sector_table_excludes_chart_geometry(sample_pdf):
    sector = _parse_page_91(sample_pdf)[1]

    assert all(len(row.raw_cells) == 3 for row in sector.rows)
    assert not any("Sector by Geography" in (cell or "") for row in sector.rows for cell in row.raw_cells)
    banks = next(row for row in sector.rows if row.row_label == "Financials - Banks")
    assert banks.raw_cells == ["Financials - Banks", "5.2", "19%"]
    assert sector.parse_warnings

