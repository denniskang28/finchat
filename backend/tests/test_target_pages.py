import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.repair_providers import (
    CellRepairProposal,
    ChartExtractionProposal,
    ChartRowProposal,
    EvidenceValue,
    RepairProposalResponse,
    TableRepairProvider,
)
from app.ingestion.renderers import build_table_debug
from app.ingestion.table_quality import TableRepairPipeline, detect_table_issues


class _GenericTargetPageProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    @property
    def available(self) -> bool:
        return True

    def propose_repairs(self, *, table, **_):
        if table.page_number != 70:
            return RepairProposalResponse()
        outer_2010 = [f"p70w{index}" for index in range(7, 12)]
        outer_2025 = [f"p70w{index}" for index in range(12, 17)]
        inner = [
            ["p70w19", "p70w20", "p70w27"],
            ["p70w17", "p70w25", "p70w32", "p70w33"],
            ["p70w21", "p70w28"],
            ["p70w22", "p70w23", "p70w29"],
            ["p70w18", "p70w26", "p70w34", "p70w35"],
            ["p70w24", "p70w30"],
        ]
        labels = [
            "As at 30 Nov 2010 Risk Discount Rates",
            "As at 30 Nov 2010 Long-term 10-year Govt Bonds",
            "As at 30 Nov 2010 Risk Premium",
            "As at 31 Dec 2025 Risk Discount Rates",
            "As at 31 Dec 2025 Long-term 10-year Govt Bonds",
            "As at 31 Dec 2025 Risk Premium",
        ]
        return RepairProposalResponse(
            repairs=[
                CellRepairProposal(
                    operation="MERGE_HEADER",
                    column_index=index,
                    value=labels[index - 1],
                    source_token_ids=[
                        *(outer_2010 if index <= 3 else outer_2025),
                        *inner[index - 1],
                    ],
                    consume_row_indices=[1],
                    reason="Merge the two visual header tiers.",
                    confidence=0.99,
                )
                for index in range(1, 7)
            ]
        )

    def propose_chart_extraction(self, **_):
        return ChartExtractionProposal(
            rows=[
                ChartRowProposal(
                    label=EvidenceValue(value="Other", source_token_ids=["p93w6"]),
                    share=EvidenceValue(value="1%", source_token_ids=["p93w9"]),
                ),
                ChartRowProposal(
                    label=EvidenceValue(
                        value="Real Estate", source_token_ids=["p93w7", "p93w8"]
                    ),
                    share=EvidenceValue(value="2%", source_token_ids=["p93w10"]),
                ),
                ChartRowProposal(
                    label=EvidenceValue(
                        value="Equities(2)", source_token_ids=["p93w14"]
                    ),
                    share=EvidenceValue(value="18%", source_token_ids=["p93w15"]),
                ),
                ChartRowProposal(
                    label=EvidenceValue(
                        value="Corporate Bonds(1)",
                        source_token_ids=["p93w30", "p93w31"],
                    ),
                    share=EvidenceValue(value="5%", source_token_ids=["p93w32"]),
                ),
                ChartRowProposal(
                    label=EvidenceValue(
                        value="Government & Government Agency Bonds",
                        source_token_ids=[
                            "p93w66",
                            "p93w67",
                            "p93w77",
                            "p93w78",
                            "p93w79",
                        ],
                    ),
                    share=EvidenceValue(value="74%", source_token_ids=["p93w80"]),
                ),
            ],
            reason="Associate the five percentages that sum to 100% with nearby labels.",
            confidence=0.99,
        )


def test_page_70_uses_generic_multilevel_header_repair(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_GenericTargetPageProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[69]
        raw = parser.parse_page(page, 70, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw)

    assert raw.table_id == "p70_t1"
    assert raw.extraction_method == "pdfplumber.lines.generic"
    assert any(issue.code == "HEADER_ROW_AS_DATA" for issue in detect_table_issues(raw))
    assert artifact.quality.status == "REPAIRED"
    assert len(artifact.canonical_table.rows) == 14
    china = next(
        row for row in artifact.canonical_table.rows if row.row_label == "Mainland China"
    )
    assert china.raw_cells[4:] == ["8.30", "2.70", "5.60"]
    semantic = next(
        row
        for row in build_table_debug(artifact).semantic_rows
        if row.row_label == "Mainland China"
    )
    assert "As at 31 Dec 2025 Risk Discount Rates: 8.30%." in semantic.content


def test_page_93_uses_generic_percentage_chart_fallback(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_GenericTargetPageProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[92]
        raw = parser.parse_page(page, 93, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw)

    assert raw.table_id == "p93_chart1"
    assert raw.source_kind == "CHART"
    assert raw.extraction_method.endswith("percentage_chart_candidate")
    assert [row.raw_cells[1] for row in raw.rows] == ["1%", "2%", "18%", "5%", "74%"]
    assert artifact.quality.status == "REPAIRED"
    assert {row.row_label: row.raw_cells[1] for row in artifact.canonical_table.rows} == {
        "Other": "1%",
        "Real Estate": "2%",
        "Equities": "18%",
        "Corporate Bonds": "5%",
        "Government & Government Agency Bonds": "74%",
    }
    assert len(artifact.repairs) == 1
    assert artifact.repairs[0].operation == "RECONSTRUCT_TABLE"


class _InventedChartValueProvider(_GenericTargetPageProvider):
    def propose_chart_extraction(self, **kwargs):
        proposal = super().propose_chart_extraction(**kwargs)
        proposal.rows[2].share.value = "19%"
        return proposal


def test_generic_chart_fallback_rejects_changed_percentage(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_InventedChartValueProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[92]
        raw = parser.parse_page(page, 93, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw)

    assert artifact.quality.status == "NEEDS_REVIEW"
    assert all(repair.operation != "RECONSTRUCT_TABLE" for repair in artifact.repairs)
    assert any(
        issue.code == "MISSING_ROW_LABEL" for issue in artifact.quality.remaining_issues
    )


def test_generic_chart_candidate_ignores_hidden_and_cross_chart_percentages(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        assert parser.parse_page(pdf.pages[1], 2, uuid.uuid4()) == []
        page_10 = parser.parse_page(pdf.pages[9], 10, uuid.uuid4())[0]

    assert [row.raw_cells[1] for row in page_10.rows] == ["5%", "7%", "29%", "59%"]
    assert sum(float(row.raw_cells[1].rstrip("%")) for row in page_10.rows) == 100
    assert page_10.bbox[2] - page_10.bbox[0] < 400
