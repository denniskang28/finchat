import uuid

import pdfplumber

from app.ingestion.parser import AiaPdfParser
from app.ingestion.repair_providers import (
    CellRepairProposal,
    DisabledTableRepairProvider,
    EvidenceValue,
    OpenAICompatibleVisionProvider,
    RepairProposalResponse,
    TableReconstructionProposal,
    TableRepairProvider,
)
from app.ingestion.renderers import build_table_debug
from app.ingestion.table_quality import TableRepairPipeline, detect_table_issues
from app.ingestion.table_quality import _apply_llm_repairs
from app.schemas import ParsedColumn, ParsedRow, ParsedTable, SourceToken


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
        for page_number, expected_issue in (
            (82, "HEADER_ROW_AS_DATA"),
            (86, "MISSING_ROW_LABEL"),
        ):
            page = pdf.pages[page_number - 1]
            table = parser.parse_page(page, page_number, uuid.uuid4())[0]
            artifact = pipeline.process(page, table)
            assert artifact.quality.status == "NEEDS_REVIEW"
            assert artifact.repairs == []
            assert any(
                issue.code == expected_issue
                for issue in artifact.quality.remaining_issues
            )


class _Page86ReconstructionProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    @property
    def available(self) -> bool:
        return True

    def propose_repairs(self, **_):
        return RepairProposalResponse()

    def propose_reconstruction(self, *, table, **_):
        token_ids = [
            (["p86w25", "p86w26", "p86w27", "p86w28", "p86w29"], ["p86w30"]),
            (["p86w38", "p86w39"], ["p86w40"]),
            (["p86w56", "p86w57"], ["p86w58"]),
            (["p86w65", "p86w66", "p86w67"], ["p86w68"]),
            (["p86w69", "p86w70"], ["p86w71"]),
            (
                [
                    "p86w72", "p86w73", "p86w74", "p86w75",
                    "p86w76", "p86w77", "p86w78", "p86w79",
                ],
                ["p86w80"],
            ),
            (["p86w94", "p86w95"], ["p86w96"]),
            (["p86w97"], ["p86w98"]),
            (["p86w104", "p86w105"], ["p86w106"]),
            (["p86w115"], ["p86w116"]),
            (["p86w125", "p86w126", "p86w127"], ["p86w128"]),
            (["p86w145", "p86w146", "p86w147", "p86w148", "p86w149"], ["p86w150"]),
        ]
        return TableReconstructionProposal(
            columns=[
                EvidenceValue(value="($b)", source_token_ids=["p86w16"]),
                EvidenceValue(
                    value="Non-par and Surplus Assets",
                    source_token_ids=["p86w8", "p86w13", "p86w14", "p86w17"],
                ),
            ],
            rows=[
                [
                    EvidenceValue(value=row.raw_cells[0], source_token_ids=ids[0]),
                    EvidenceValue(value=row.raw_cells[1], source_token_ids=ids[1]),
                ]
                for row, ids in zip(table.rows, token_ids, strict=True)
            ],
            reason="The right-side chart was joined to a dense two-column table.",
            confidence=0.99,
        )


def test_page_86_llm_reconstructs_only_dense_token_verified_table(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_Page86ReconstructionProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[85]
        raw_table = parser.parse_page(page, 86, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw_table)

    assert len(raw_table.columns) == 5
    assert len(raw_table.rows) == 13
    assert artifact.quality.status == "REPAIRED"
    assert artifact.quality.remaining_issues == []
    assert [column.label for column in artifact.canonical_table.columns] == [
        "($b)",
        "Non-par and Surplus Assets",
    ]
    assert len(artifact.canonical_table.rows) == 12
    assert artifact.canonical_table.rows[-2].raw_cells == [
        "Total Invested Assets",
        "149.1",
    ]
    assert artifact.canonical_table.rows[-1].raw_cells == [
        "% of Total Invested Assets",
        "52%",
    ]
    assert artifact.repairs[-1].operation == "RECONSTRUCT_TABLE"
    assert len(artifact.repairs[-1].source_token_ids) == 53
    assert artifact.canonical_table.bbox[2] < raw_table.bbox[2]


class _InventedValueReconstructionProvider(_Page86ReconstructionProvider):
    def propose_reconstruction(self, **kwargs):
        proposal = super().propose_reconstruction(**kwargs)
        proposal.rows[-1][1].value = "53%"
        return proposal


def test_page_86_reconstruction_rejects_an_llm_changed_number(sample_pdf):
    parser = AiaPdfParser()
    pipeline = TableRepairPipeline(_InventedValueReconstructionProvider())
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[85]
        raw_table = parser.parse_page(page, 86, uuid.uuid4())[0]
        artifact = pipeline.process(page, raw_table)

    assert artifact.quality.status == "NEEDS_REVIEW"
    assert all(repair.operation != "RECONSTRUCT_TABLE" for repair in artifact.repairs)
    assert len(artifact.canonical_table.columns) == 5
    assert any(
        issue.code == "OVERSIZED_CELL" for issue in artifact.quality.remaining_issues
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
    assert artifact.quality.status == "REPAIRED"


def _synthetic_multilevel_header_table() -> ParsedTable:
    columns = [
        ParsedColumn(
            key="currency",
            source_labels=["%"],
            label="%",
            semantic_label="%",
            is_row_label=True,
            metadata={"source_x_range": [0, 100], "header_y_range": [0, 20]},
        ),
        ParsedColumn(
            key="year_1",
            source_labels=["1 year"],
            label="1 year",
            semantic_label="1 year",
            metadata={"source_x_range": [100, 200], "header_y_range": [0, 20]},
        ),
        ParsedColumn(
            key="column_3",
            source_labels=["Column 3"],
            label="Column 3",
            semantic_label="Column 3",
            metadata={"source_x_range": [200, 300], "header_y_range": [0, 20]},
        ),
        ParsedColumn(
            key="column_4",
            source_labels=["Column 4"],
            label="Column 4",
            semantic_label="Column 4",
            metadata={"source_x_range": [300, 400], "header_y_range": [0, 20]},
        ),
    ]
    rows = [
        ParsedRow(
            row_index=1,
            row_label="Row 1",
            values={
                "currency": None,
                "year_1": "Risk free",
                "column_3": "With illiquidity premium",
                "column_4": None,
            },
            display_values={
                "currency": None,
                "year_1": "Risk free",
                "column_3": "With illiquidity premium",
                "column_4": None,
            },
            raw_cells=[None, "Risk free", "With illiquidity premium", None],
            metadata={"source_row_bbox": [0, 20, 400, 50], "label_x_range": [0, 100]},
        ),
        ParsedRow(
            row_index=2,
            row_label="USD",
            values={"currency": "USD", "year_1": "3.43", "column_3": "3.93", "column_4": None},
            display_values={"currency": "USD", "year_1": "3.43", "column_3": "3.93", "column_4": None},
            raw_cells=["USD", "3.43", "3.93", None],
            metadata={"source_row_bbox": [0, 50, 400, 80], "label_x_range": [0, 100]},
        ),
    ]
    return ParsedTable(
        table_id="synthetic",
        document_id=uuid.uuid4(),
        page_number=1,
        bbox=(0, 0, 400, 80),
        title="Synthetic",
        columns=columns,
        rows=rows,
        extraction_method="test",
    )


def test_llm_can_merge_multilevel_headers_and_delete_verified_empty_column():
    table = _synthetic_multilevel_header_table()
    tokens = [
        SourceToken(token_id="year", text="1 year", bbox=(125, 5, 175, 15)),
        SourceToken(token_id="risk", text="Risk free", bbox=(115, 25, 180, 40)),
        SourceToken(token_id="with", text="With illiquidity premium", bbox=(205, 25, 290, 40)),
    ]
    proposals = RepairProposalResponse(
        repairs=[
            CellRepairProposal(
                operation="MERGE_HEADER",
                column_index=1,
                value="1 year / Risk free",
                source_token_ids=["year", "risk"],
                consume_row_indices=[1],
                reason="Merge two header tiers.",
                confidence=0.98,
            ),
            CellRepairProposal(
                operation="MERGE_HEADER",
                column_index=2,
                value="1 year / With illiquidity premium",
                source_token_ids=["year", "with"],
                consume_row_indices=[1],
                reason="Merge two header tiers.",
                confidence=0.98,
            ),
            CellRepairProposal(
                operation="DELETE_EMPTY_COLUMN",
                column_index=3,
                reason="The placeholder column is empty.",
                confidence=0.99,
            ),
        ]
    )

    repairs = _apply_llm_repairs(
        table=table,
        provider=_EvidenceBackedFakeProvider(),
        proposals=proposals,
        tokens=tokens,
    )

    assert [column.label for column in table.columns] == [
        "%",
        "1 year / Risk free",
        "1 year / With illiquidity premium",
    ]
    assert [row.row_label for row in table.rows] == ["USD"]
    assert {repair.operation for repair in repairs} == {
        "MERGE_HEADER",
        "DELETE_EMPTY_COLUMN",
    }


def test_llm_can_trim_contamination_but_cannot_invent_replacement_text():
    table = _synthetic_multilevel_header_table()
    table.rows[1].raw_cells[1] = "3.43 nearby chart narrative"
    table.rows[1].values["year_1"] = "3.43 nearby chart narrative"
    table.rows[1].display_values["year_1"] = "3.43 nearby chart narrative"
    tokens = [SourceToken(token_id="value", text="3.43", bbox=(120, 55, 150, 70))]
    proposals = RepairProposalResponse(
        repairs=[
            CellRepairProposal(
                operation="TRIM_CONTAMINATED_CELL",
                row_index=2,
                column_index=1,
                value="3.43",
                source_token_ids=["value"],
                reason="Remove unrelated chart narrative.",
                confidence=0.97,
            ),
            CellRepairProposal(
                operation="REPLACE_HEADER",
                column_index=3,
                value="Invented header",
                source_token_ids=[],
                reason="No source evidence.",
                confidence=0.99,
            ),
        ]
    )

    repairs = _apply_llm_repairs(
        table=table,
        provider=_EvidenceBackedFakeProvider(),
        proposals=proposals,
        tokens=tokens,
    )

    assert table.rows[1].raw_cells[1] == "3.43"
    assert table.columns[3].label == "Column 4"
    assert [repair.operation for repair in repairs] == ["TRIM_CONTAMINATED_CELL"]


def test_llm_rejects_duplicate_header_and_reassigns_source_aligned_token():
    table = _synthetic_multilevel_header_table()
    tokens = [
        SourceToken(token_id="duplicate", text="1 year", bbox=(210, 5, 260, 15)),
        SourceToken(token_id="usd", text="USD", bbox=(10, 55, 45, 70)),
    ]
    proposals = RepairProposalResponse(
        repairs=[
            CellRepairProposal(
                operation="MERGE_HEADER",
                column_index=2,
                value="1 year",
                source_token_ids=["duplicate"],
                reason="Would duplicate an existing header.",
                confidence=0.99,
            ),
            CellRepairProposal(
                operation="REASSIGN_TOKEN",
                row_index=2,
                column_index=3,
                value="USD",
                source_token_ids=["usd"],
                source_row_index=2,
                source_column_index=0,
                reason="Move the token from its extracted source cell.",
                confidence=0.98,
            ),
        ]
    )

    repairs = _apply_llm_repairs(
        table=table,
        provider=_EvidenceBackedFakeProvider(),
        proposals=proposals,
        tokens=tokens,
    )

    assert table.columns[2].label == "Column 3"
    assert table.rows[1].raw_cells == [None, "3.43", "3.93", "USD"]
    assert [repair.operation for repair in repairs] == ["REASSIGN_TOKEN"]


def test_provider_uses_compact_non_thinking_prompt_and_skips_invalid_proposals(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"repairs": ['
                                '{"operation":"UNSUPPORTED","column_index":1},'
                                '{"operation":"DELETE_EMPTY_COLUMN","column_index":3,'
                                '"reason":"Empty placeholder","confidence":0.99}'
                                "]}"
                            )
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr("app.ingestion.repair_providers.httpx.post", fake_post)
    provider = OpenAICompatibleVisionProvider(
        name="fake",
        api_key="not-a-real-key",
        base_url="https://example.invalid/v1",
        model="qwen3.8-flash",
        timeout_seconds=1,
    )
    table = _synthetic_multilevel_header_table()
    response = provider.propose_repairs(
        table=table,
        issues=detect_table_issues(table),
        tokens=[],
        image_data_url="data:image/png;base64,AA==",
    )

    assert captured["enable_thinking"] is False
    assert captured["max_tokens"] == 3000
    assert len(response.repairs) == 1
    assert response.repairs[0].operation == "DELETE_EMPTY_COLUMN"
