import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    filename = "AIA Group 2025 Annual Results Analyst Presentation EN (FINAL) - 27 Mar.pdf"
    candidates = [
        Path(os.environ.get("SAMPLE_PDF_PATH", "")),
        Path(__file__).resolve().parents[2] / "docs" / "samples" / filename,
        Path("/app/docs/samples") / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Sample PDF not found; checked: {candidates}")

