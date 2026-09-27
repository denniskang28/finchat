import uuid

import pdfplumber

from app.ingestion.parser import PdfParser
from app.ingestion.repair_providers import (
    ContentFactProposal,
    ContentSectionProposal,
    EvidenceValue,
    PageStructureResponse,
    RepairProposalResponse,
    TableRepairProvider,
)
from app.ingestion.table_quality import TableRepairPipeline


def _evidence(value, *token_ids):
    return EvidenceValue(value=value, source_token_ids=list(token_ids))


class _Page5ContentProvider(TableRepairProvider):
    name = "fake"
    model = "fake-vision"

    def __init__(self, *, include_silver_segment=True):
        self.include_silver_segment = include_silver_segment

    @property
    def available(self):
        return True

    def propose_repairs(self, **_):
        return RepairProposalResponse()

    def propose_page_structures(self, **_):
        demographic_facts = [
            ContentFactProposal(
                label=_evidence("Large Working Age Population", "p5w48", "p5w49", "p5w50", "p5w51"),
                value=_evidence("2.6b", "p5w57"),
            ),
            ContentFactProposal(
                label=_evidence("Age 15-64 in 2030E(1)", "p5w63", "p5w64", "p5w65", "p5w66"),
            ),
        ]
        if self.include_silver_segment:
            demographic_facts.extend([
                ContentFactProposal(
                    label=_evidence("Growing Silver Segment", "p5w52", "p5w53", "p5w54"),
                    value=_evidence("700m", "p5w58"),
                ),
                ContentFactProposal(
                    label=_evidence("Age 60+ in 2030E(1)", "p5w67", "p5w68", "p5w69", "p5w70"),
                ),
            ])
        return PageStructureResponse(
            content_sections=[
                ContentSectionProposal(
                    heading_path=[
                        _evidence("Structural Growth Drivers", "p5w21", "p5w22", "p5w23", "p5w24", "p5w25"),
                        _evidence("Favourable Demographics", "p5w37", "p5w38"),
                    ],
                    facts=demographic_facts,
                    reason="Group the two demographic KPI cards.",
                    confidence=0.99,
                ),
                ContentSectionProposal(
                    heading_path=[
                        _evidence("Unparalleled Opportunities", "p5w26", "p5w27"),
                        _evidence(
                            "Large Protection Gap Driving Life and Health Insurance Demand",
                            "p5w39", "p5w40", "p5w41", "p5w42", "p5w43", "p5w44", "p5w45", "p5w46", "p5w47",
                        ),
                    ],
                    facts=[
                        ContentFactProposal(
                            label=_evidence("Mortality Protection Gap(4)", "p5w71", "p5w72", "p5w73"),
                            value=_evidence(">$130b", "p5w55"),
                        ),
                        ContentFactProposal(
                            label=_evidence("Health Protection Gap(4)", "p5w74", "p5w75", "p5w76"),
                            value=_evidence(">$250b", "p5w56"),
                        ),
                    ],
                    reason="Group the two protection-gap KPI cards.",
                    confidence=0.99,
                ),
                ContentSectionProposal(
                    heading_path=[
                        _evidence("Structural Growth Drivers", "p5w21", "p5w22", "p5w23", "p5w24", "p5w25"),
                        _evidence("Large Personal Health Expenditure", "p5w133", "p5w134", "p5w135", "p5w136"),
                    ],
                    facts=[
                        ContentFactProposal(
                            label=_evidence(
                                "Annual healthcare expenditure across Asia ex-Japan(2)",
                                "p5w151", "p5w152", "p5w167", "p5w168", "p5w177", "p5w178",
                            ),
                            value=_evidence(">$1.7t", "p5w147"),
                        ),
                        ContentFactProposal(
                            label=_evidence(
                                "of total healthcare expenditure is out-of-pocket(3)",
                                "p5w153", "p5w154", "p5w155", "p5w169", "p5w170", "p5w179",
                            ),
                            value=_evidence("42%", "p5w148"),
                        ),
                    ],
                    reason="Group the two health-expenditure KPI cards.",
                    confidence=0.99,
                ),
                ContentSectionProposal(
                    heading_path=[
                        _evidence("Unparalleled Opportunities", "p5w26", "p5w27"),
                        _evidence(
                            "Need for Personalised Financial Advice",
                            "p5w137", "p5w138", "p5w139", "p5w140", "p5w141",
                        ),
                    ],
                    facts=[
                        ContentFactProposal(
                            label=_evidence(
                                "want access to a human adviser when buying life or health insurance(5)",
                                "p5w156", "p5w157", "p5w158", "p5w159", "p5w160", "p5w171", "p5w172", "p5w173", "p5w174", "p5w180", "p5w181", "p5w182",
                            ),
                            value=_evidence("98%", "p5w149"),
                        ),
                        ContentFactProposal(
                            label=_evidence(
                                "want empathetic, personalised engagement from insurance agents(6)",
                                "p5w161", "p5w162", "p5w175", "p5w176", "p5w183", "p5w184", "p5w185",
                            ),
                            value=_evidence("83%", "p5w150"),
                        ),
                    ],
                    reason="Group the two advice-preference KPI cards.",
                    confidence=0.99,
                ),
                ContentSectionProposal(
                    heading_path=[
                        _evidence(
                            "AIA’s Key Competitive Advantages to Meet Evolving Customer Needs",
                            "p5w28", "p5w29", "p5w30", "p5w31", "p5w32", "p5w33", "p5w34", "p5w35", "p5w36",
                        )
                    ],
                    facts=[
                        ContentFactProposal(label=_evidence("Unrivalled Proprietary Premier Agency", "p5w59", "p5w60", "p5w61", "p5w62")),
                        ContentFactProposal(label=_evidence("Long-term Strategic Partnerships", "p5w86", "p5w87", "p5w88")),
                        ContentFactProposal(label=_evidence("Leading Customer Experience", "p5w109", "p5w110", "p5w111")),
                        ContentFactProposal(
                            label=_evidence(
                                "Compelling Propositions with Integrated Products and Services",
                                "p5w131", "p5w132", "p5w142", "p5w143", "p5w144", "p5w145", "p5w146",
                            )
                        ),
                        ContentFactProposal(label=_evidence("Industry-leading Technology and AI", "p5w163", "p5w164", "p5w165", "p5w166")),
                    ],
                    reason="Keep the five qualitative competitive advantages together.",
                    confidence=0.99,
                ),
            ]
        )


def _parse_page_5(sample_pdf, provider):
    parser = PdfParser()
    pipeline = TableRepairPipeline(provider, llm_primary=True)
    with pdfplumber.open(sample_pdf) as pdf:
        page = pdf.pages[4]
        page_text = page.extract_text(layout=True) or ""
        raw_tables = parser.parse_page(page, 5, uuid.uuid4(), page_text)
        return pipeline.process_page_with_content(
            page=page,
            raw_tables=raw_tables,
            document_id=uuid.uuid4(),
            page_number=5,
            page_title=parser._page_title(page_text),
            footnotes=parser._note_lines(page_text, 5),
        )


def test_page_5_builds_hierarchical_sections_and_atomic_facts(sample_pdf):
    _, content = _parse_page_5(sample_pdf, _Page5ContentProvider())

    assert len(content.sections) == 5
    facts = {fact.label: fact for section in content.sections for fact in section.facts}
    working_age = "Large Working Age Population - Age 15-64 in 2030E(1)"
    silver_segment = "Growing Silver Segment - Age 60+ in 2030E(1)"
    assert facts[working_age].value == "2.6b"
    assert facts[silver_segment].value == "700m"
    assert facts["Mortality Protection Gap(4)"].value == ">$130b"
    assert facts["Health Protection Gap(4)"].value == ">$250b"
    assert facts["Industry-leading Technology and AI"].value is None
    assert facts[working_age].footnotes == ["(1) AIA markets only"]
    assert facts["Mortality Protection Gap(4)"].footnotes == [
        "(4) Premium equivalent in 2024"
    ]
    assert "Structural Growth Drivers > Favourable Demographics" in facts[working_age].content
    assert "AIA’s Key Competitive Advantages" in content.summary
    assert facts[working_age].source_token_ids == [
        "p5w48", "p5w49", "p5w50", "p5w51", "p5w57",
        "p5w63", "p5w64", "p5w65", "p5w66",
    ]


def test_page_5_coverage_reports_an_unassigned_kpi(sample_pdf):
    _, content = _parse_page_5(
        sample_pdf,
        _Page5ContentProvider(include_silver_segment=False),
    )

    uncovered = {token.token_id: token.text for token in content.coverage.uncovered_tokens}
    assert uncovered["p5w58"] == "700m"
    assert any("numeric tokens" in warning for warning in content.coverage.warnings)
