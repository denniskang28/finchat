import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.renderers import build_table_debug


def test_page_70_multilevel_headers_are_flattened(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        table = parser.parse_page(pdf.pages[69], 70, uuid.uuid4())[0]
    china = next(row for row in table.rows if row.row_label == "Mainland China")

    assert len(table.columns) == 7
    assert china.display_values["rdr_2025"] == "8.30"
    assert china.display_values["risk_premium_2025"] == "5.60"
    semantic = next(
        row for row in build_table_debug(table).semantic_rows if row.row_label == "Mainland China"
    )
    assert "As at 31 Dec 2025, Risk discount rate: 8.30%." in semantic.content


def test_page_93_targeted_chart_adapter(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        table = parser.parse_page(pdf.pages[92], 93, uuid.uuid4())[0]

    assert table.source_kind == "CHART"
    assert {row.row_label: row.raw_cells[1] for row in table.rows} == {
        "Real Estate": "2%",
        "Other": "1%",
        "Equities": "18%",
        "Corporate Bonds": "5%",
        "Government & Government Agency Bonds": "74%",
    }
    assert not table.parse_warnings

