import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.repair_providers import (
    EvidenceValue,
    RepairProposalResponse,
    TableReconstructionProposal,
    TableRepairProvider,
)
from app.ingestion.renderers import build_table_debug, render_table_summary
from app.ingestion.table_quality import TableRepairPipeline


class _Page91GenericProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    @property
    def available(self) -> bool:
        return True

    def propose_repairs(self, **_):
        return RepairProposalResponse()

    def propose_reconstruction(self, *, table, **_):
        label_ids = [
            ["p91w36"],
            ["p91w43"],
            ["p91w49"],
            ["p91w55", "p91w56"],
            ["p91w65", "p91w66"],
            ["p91w73"],
            ["p91w77", "p91w78", "p91w79"],
            ["p91w84", "p91w85", "p91w86", "p91w87"],
            ["p91w91", "p91w92", "p91w93"],
            ["p91w97", "p91w98"],
            ["p91w103", "p91w104"],
            ["p91w107", "p91w108"],
            ["p91w111"],
            ["p91w114"],
        ]
        amount_ids = [
            "p91w37", "p91w44", "p91w50", "p91w57", "p91w67", "p91w74",
            "p91w80", "p91w88", "p91w94", "p91w99", "p91w105", "p91w109",
            "p91w112", "p91w115",
        ]
        share_ids = [
            "p91w38", "p91w45", "p91w51", "p91w58", "p91w68", "p91w75",
            "p91w81", "p91w89", "p91w95", "p91w100", "p91w106", "p91w110",
            "p91w113", "p91w116",
        ]
        return TableReconstructionProposal(
            columns=[
                EvidenceValue(value="Row label"),
                EvidenceValue(value="$b", source_token_ids=["p91w28"]),
                EvidenceValue(
                    value="% of total",
                    source_token_ids=["p91w29", "p91w30", "p91w31"],
                ),
            ],
            rows=[
                [
                    EvidenceValue(value=row.raw_cells[0], source_token_ids=label_ids[index]),
                    EvidenceValue(value=row.raw_cells[1], source_token_ids=[amount_ids[index]]),
                    EvidenceValue(value=row.raw_cells[2], source_token_ids=[share_ids[index]]),
                ]
                for index, row in enumerate(table.rows)
            ],
            reason="Remove sparse chart geometry to the right of the dense table.",
            confidence=0.99,
        )


def _parse_page_91(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_Page91GenericProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[90]
        return [
            pipeline.process(page, table)
            for table in parser.parse_page(page, 91, uuid.uuid4())
        ]


def test_page_91_uses_two_generic_tables(sample_pdf):
    artifacts = _parse_page_91(sample_pdf)
    tables = [artifact.canonical_table for artifact in artifacts]

    assert [table.table_id for table in tables] == ["p91_t1", "p91_t2"]
    assert [table.title for table in tables] == [
        "Corporate Bonds by Geography - Table 1",
        "Corporate Bonds by Sector - Table 2",
    ]
    assert [len(table.columns) for table in tables] == [3, 3]
    assert [len(table.rows) for table in tables] == [4, 14]
    assert [artifact.quality.status for artifact in artifacts] == ["PASS", "REPAIRED"]
    assert artifacts[1].repairs[0].operation == "RECONSTRUCT_TABLE"


def test_page_91_united_states_values_and_representations(sample_pdf):
    geography = _parse_page_91(sample_pdf)[0].canonical_table
    united_states = next(row for row in geography.rows if row.row_label == "United States")

    assert united_states.raw_cells == ["United States", "6.2", "22%"]
    assert united_states.display_values["b"] == "6.2"
    assert united_states.display_values["of_total"] == "22%"

    debug = build_table_debug(geography)
    raw = next(row for row in debug.raw_rows if row.row_label == "United States")
    semantic = next(row for row in debug.semantic_rows if row.row_label == "United States")

    assert raw.comparison_key == "p91_t1:united_states"
    assert raw.content == "United States | 6.2 | 22%"
    assert "| United States | 6.2 | 22% |" in debug.markdown
    assert "$b: USD 6.2 billion." in semantic.content
    assert "% of total: 22%." in semantic.content
    summary = render_table_summary(geography)
    assert "Table: Corporate Bonds by Geography - Table 1." in summary
    assert "Columns: Row label; $b (USD billion); % of total (%)." in summary
    assert "Rows: Asia Pacific; United States; Other; Total." in summary


def test_page_91_generic_reconstruction_excludes_chart_geometry(sample_pdf):
    artifact = _parse_page_91(sample_pdf)[1]
    sector = artifact.canonical_table

    assert all(len(row.raw_cells) == 3 for row in sector.rows)
    assert not any(
        "Sector by Geography" in (cell or "")
        for row in sector.rows
        for cell in row.raw_cells
    )
    banks = next(row for row in sector.rows if row.row_label == "Financials - Banks")
    assert banks.raw_cells == ["Financials - Banks", "5.2", "19%"]
    assert artifact.quality.remaining_issues == []
