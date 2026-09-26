import uuid
from pathlib import Path

import pdfplumber
import pytest

from app.ingestion.parser import PdfParser
from app.ingestion.repair_providers import (
    EvidenceValue,
    PageColumnProposal,
    PageStructureProposal,
    PageStructureResponse,
    RepairProposalResponse,
    TableRepairProvider,
)
from app.ingestion.table_quality import TableRepairPipeline
from app.ingestion.table_quality import _validated_page_structure
from app.schemas import SourceToken


@pytest.fixture(scope="session")
def sample_2024_pdf() -> Path:
    filename = "AIA Group 2024 Annual Results Analyst Presentation (Final) (1).pdf"
    candidates = [
        Path(__file__).resolve().parents[2] / "docs" / "samples" / filename,
        Path("/app/docs/samples") / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Sample PDF not found; checked: {candidates}")


class _Page60PrimaryProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    @property
    def available(self) -> bool:
        return True

    def propose_repairs(self, **_):
        return RepairProposalResponse()

    def propose_page_structures(self, **_):
        row_token_ids = [
            (["p60w25", "p60w26", "p60w27", "p60w28", "p60w29"], ["p60w30"]),
            (["p60w38", "p60w39"], ["p60w40"]),
            (["p60w56", "p60w57"], ["p60w58"]),
            (["p60w65", "p60w66", "p60w67"], ["p60w68"]),
            (["p60w69", "p60w70"], ["p60w71"]),
            (
                [
                    "p60w72", "p60w73", "p60w74", "p60w75",
                    "p60w76", "p60w77", "p60w78", "p60w79",
                ],
                ["p60w80"],
            ),
            (["p60w94", "p60w95"], ["p60w96"]),
            (["p60w97"], ["p60w98"]),
            (["p60w104", "p60w105"], ["p60w106"]),
            (["p60w115"], ["p60w116"]),
            (["p60w125", "p60w126", "p60w127"], ["p60w128"]),
            (["p60w145", "p60w146", "p60w147", "p60w148", "p60w149"], ["p60w150"]),
        ]
        values = [
            ("Government & Government Agency Bonds", "73.4"),
            ("Corporate Bonds", "28.7"),
            ("Structured Securities", "1.8"),
            ("Loans and Deposits", "3.6"),
            ("Fixed Income", "107.5"),
            ("Interests in investment funds & exchangeable loan notes", "10.6"),
            ("Equity shares", "5.3"),
            ("Equities(1)", "15.8"),
            ("Real Estate", "4.8"),
            ("Others(2)", "6.1"),
            ("Total Invested Assets", "134.1"),
            ("% of Total Invested Assets", "53%"),
        ]
        rows = [
            [
                EvidenceValue(value="($b)", source_token_ids=["p60w17"]),
                EvidenceValue(value=""),
            ],
            *[
                [
                    EvidenceValue(value=values[index][0], source_token_ids=token_ids[0]),
                    EvidenceValue(value=values[index][1], source_token_ids=token_ids[1]),
                ]
                for index, token_ids in enumerate(row_token_ids)
            ],
        ]
        return PageStructureResponse(
            structures=[
                PageStructureProposal(
                    source_kind="TABLE",
                    columns=[
                        PageColumnProposal(
                            label=EvidenceValue(value="Row label"),
                            is_row_label=True,
                        ),
                        PageColumnProposal(
                            label=EvidenceValue(
                                value="Non-par and Surplus Assets",
                                source_token_ids=["p60w8", "p60w13", "p60w14", "p60w16"],
                            ),
                            value_type="CURRENCY",
                        ),
                    ],
                    rows=rows,
                    reason="Separate the dense left-hand table from the adjacent chart.",
                    confidence=0.99,
                )
            ]
        )


def _parse_page_60(sample_2024_pdf: Path, provider: TableRepairProvider):
    parser = PdfParser()
    pipeline = TableRepairPipeline(provider, llm_primary=True)
    with pdfplumber.open(sample_2024_pdf) as pdf:
        page = pdf.pages[59]
        page_text = page.extract_text(layout=True) or ""
        raw_tables = parser.parse_page(page, 60, uuid.uuid4(), page_text)
        return pipeline.process_page(
            page=page,
            raw_tables=raw_tables,
            document_id=raw_tables[0].document_id,
            page_number=60,
            page_title=parser._page_title(page_text),
            footnotes=parser._note_lines(page_text),
        )


def test_llm_primary_normalizes_page_60_unit_row(sample_2024_pdf):
    artifacts = _parse_page_60(sample_2024_pdf, _Page60PrimaryProvider())

    assert len(artifacts) == 1
    table = artifacts[0].canonical_table
    assert table.extraction_method == "llm.page_structure+pdfplumber.token_verified"
    assert [column.label for column in table.columns] == [
        "Row label",
        "Non-par and Surplus Assets",
    ]
    assert table.columns[1].unit == "USD billion"
    assert len(table.rows) == 12
    assert table.rows[0].raw_cells == ["Government & Government Agency Bonds", "73.4"]
    assert table.rows[-1].raw_cells == ["% of Total Invested Assets", "53%"]
    assert artifacts[0].quality.status == "PASS"
    assert artifacts[0].quality.remaining_issues == []


class _ChangedNumberProvider(_Page60PrimaryProvider):
    def propose_page_structures(self, **kwargs):
        response = super().propose_page_structures(**kwargs)
        response.structures[0].rows[-1][1].value = "54%"
        return response


def test_llm_primary_rejects_changed_number_and_falls_back(sample_2024_pdf):
    artifacts = _parse_page_60(sample_2024_pdf, _ChangedNumberProvider())

    assert len(artifacts) == 1
    table = artifacts[0].canonical_table
    assert table.extraction_method.startswith("pdfplumber.lines.generic")
    assert any("value differs from source tokens: 54%" in warning for warning in table.parse_warnings)
    assert artifacts[0].quality.status == "NEEDS_REVIEW"


class _UnavailablePageResponseProvider(_Page60PrimaryProvider):
    def propose_page_structures(self, **_):
        raise TimeoutError("simulated provider timeout")


def test_llm_primary_records_provider_failure_before_fallback(sample_2024_pdf):
    artifacts = _parse_page_60(sample_2024_pdf, _UnavailablePageResponseProvider())

    assert any(
        "provider request failed: TimeoutError" in warning
        for warning in artifacts[0].canonical_table.parse_warnings
    )


def test_chart_allows_only_token_supported_inferred_percentage_schema():
    proposal = PageStructureProposal(
        source_kind="CHART",
        columns=[
            PageColumnProposal(
                label=EvidenceValue(value="Category"),
                is_row_label=True,
            ),
            PageColumnProposal(
                label=EvidenceValue(value="Percentage"),
                unit="%",
                value_type="PERCENTAGE",
            ),
        ],
        rows=[
            [
                EvidenceValue(value="Alpha", source_token_ids=["label-a"]),
                EvidenceValue(value="40%", source_token_ids=["share-a"]),
            ],
            [
                EvidenceValue(value="Beta", source_token_ids=["label-b"]),
                EvidenceValue(value="60%", source_token_ids=["share-b"]),
            ],
        ],
        reason="Two-part distribution chart.",
        confidence=0.99,
    )
    tokens = [
        SourceToken(token_id="label-a", text="Alpha", bbox=(10, 10, 30, 20)),
        SourceToken(token_id="share-a", text="40%", bbox=(40, 10, 55, 20)),
        SourceToken(token_id="label-b", text="Beta", bbox=(10, 30, 30, 40)),
        SourceToken(token_id="share-b", text="60%", bbox=(40, 30, 55, 40)),
    ]

    table = _validated_page_structure(
        proposal=proposal,
        tokens=tokens,
        document_id=uuid.uuid4(),
        page_number=1,
        table_id="p1_chart1",
        default_title="Distribution",
        footnotes=[],
    )

    assert proposal.columns[1].value_type == "PERCENT"
    assert table.columns[1].unit == "%"
    assert [row.raw_cells for row in table.rows] == [["Alpha", "40%"], ["Beta", "60%"]]
