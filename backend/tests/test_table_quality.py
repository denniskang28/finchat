import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.repair_providers import (
    CellRepairProposal,
    DisabledTableRepairProvider,
    RepairProposalResponse,
    TableRepairProvider,
)
from app.ingestion.renderers import build_table_debug
from app.ingestion.table_quality import TableRepairPipeline, detect_table_issues


def test_page_77_missing_labels_are_detected_repaired_and_auditable(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(DisabledTableRepairProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[76]
        raw_table = parser.parse_page(page, 77, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw_table)

    assert [issue.code for issue in detect_table_issues(raw_table)] == [
        "MISSING_ROW_LABEL",
        "MISSING_ROW_LABEL",
    ]
    assert raw_table.rows[2].raw_cells == [None, "25", "(155)", "n/m"]
    assert raw_table.rows[3].raw_cells == [None, "(825)", "819", "n/m"]

    canonical = artifact.canonical_table
    assert canonical.rows[2].row_label == (
        "Reclassification of revaluation losses/(gains) for property held for own use, net of tax"
    )
    assert canonical.rows[3].row_label == "Other non-operating items, net of tax"
    assert artifact.quality.status == "REPAIRED"
    assert artifact.quality.remaining_issues == []
    assert len(artifact.repairs) == 2
    assert all(repair.source == "DETERMINISTIC" for repair in artifact.repairs)
    assert all(repair.source_token_ids for repair in artifact.repairs)

    debug = build_table_debug(artifact)
    assert debug.raw_parsed_table.rows[2].raw_cells[0] is None
    assert debug.parsed_table.rows[2].raw_cells[0].startswith("Reclassification")
    assert "| Reclassification of revaluation" in debug.markdown
    assert "Row 3" not in debug.semantic_rows[2].content


def test_page_77_reconciliation_values_remain_unchanged(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(DisabledTableRepairProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[76]
        raw_table = parser.parse_page(page, 77, uuid.uuid4())[0]
        canonical = pipeline.process(page, raw_table).canonical_table

    assert [row.raw_cells[1:] for row in canonical.rows] == [
        ["7,136", "6,658", "+7%"],
        ["(102)", "(435)", "(77)%"],
        ["25", "(155)", "n/m"],
        ["(825)", "819", "n/m"],
        ["6,234", "6,887", "(9)%"],
    ]
    assert 7136 - 102 + 25 - 825 == 6234
    assert 6658 - 435 - 155 + 819 == 6887


def test_page_78_superscript_footnote_keeps_visual_word_order(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(DisabledTableRepairProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[77]
        raw_tables = parser.parse_page(page, 78, uuid.uuid4())
        artifacts = [pipeline.process(page, table) for table in raw_tables]

    assert [row.row_label for row in artifacts[0].canonical_table.rows[1:4]] == [
        "Operating Tax",
        "Tax other than GMT Top-up tax(2)",
        "GMT Top-up Tax",
    ]
    assert [row.row_label for row in artifacts[1].canonical_table.rows[1:4]] == [
        "Tax",
        "Tax other than GMT Top-up tax",
        "GMT Top-up Tax",
    ]


def test_ambiguous_missing_labels_are_not_auto_repaired(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(DisabledTableRepairProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        for page_number in (82, 86):
            page = pdf.pages[page_number - 1]
            table = parser.parse_page(page, page_number, uuid.uuid4())[0]
            artifact = pipeline.process(page, table)
            assert artifact.quality.status == "NEEDS_REVIEW"
            assert artifact.repairs == []
            assert any(
                issue.code == "MISSING_ROW_LABEL"
                for issue in artifact.quality.remaining_issues
            )


class _EvidenceBackedFakeProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    @property
    def available(self) -> bool:
        return True

    def propose_repairs(self, *, tokens, **_):
        selected = [
            token
            for token in tokens
            if 289 < (token.bbox[1] + token.bbox[3]) / 2 < 349 and token.bbox[0] < 498
        ]
        return RepairProposalResponse(
            repairs=[
                CellRepairProposal(
                    row_index=3,
                    column_index=0,
                    value=(
                        "Reclassification of revaluation losses/(gains) for property held for own use, net of tax"
                    ),
                    source_token_ids=[token.token_id for token in selected],
                    reason="The source tokens are visually aligned with row 3.",
                    confidence=0.96,
                )
            ]
        )


def test_llm_proposal_is_applied_only_with_source_token_evidence(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[76]
        raw_table = parser.parse_page(page, 77, uuid.uuid4())[0]
        raw_table.rows[2].metadata = {}
        artifact = TableRepairPipeline(_EvidenceBackedFakeProvider()).process(page, raw_table)

    llm_repair = next(repair for repair in artifact.repairs if repair.source == "LLM")
    assert llm_repair.provider == "fake"
    assert llm_repair.model == "fake-vision"
    assert llm_repair.source_token_ids
    assert artifact.canonical_table.rows[2].row_label.startswith("Reclassification")
    assert artifact.quality.status == "REPAIRED"


def test_page_94_split_total_cell_is_merged_into_previous_row(sample_pdf):
    parser = AiaPdfParser()
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[93]
        raw_table = parser.parse_page(page, 94, uuid.uuid4())[0]
        artifact = TableRepairPipeline(DisabledTableRepairProvider()).process(page, raw_table)

    assert raw_table.rows[2].raw_cells == ["Private Credit Funds", "2.8", "3.3", None]
    assert raw_table.rows[3].raw_cells == [None, None, None, "6.1"]
    assert artifact.canonical_table.rows[2].raw_cells == [
        "Private Credit Funds",
        "2.8",
        "3.3",
        "6.1",
    ]
    assert len(artifact.canonical_table.rows) == len(raw_table.rows) - 1
    assert artifact.repairs[0].operation == "MERGE_SPLIT_ROW"
    assert artifact.quality.status == "NEEDS_REVIEW"
