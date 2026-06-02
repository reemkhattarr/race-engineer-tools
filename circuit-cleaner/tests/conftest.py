"""Shared fixtures for circuit-cleaner tests."""

from pathlib import Path

import pytest

_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "assets" / "samples"


@pytest.fixture
def sample_image() -> Path:
    """Path to a known-good sample circuit map shipped with the repo."""
    path = _SAMPLES_DIR / "a.jpeg"
    if not path.exists():
        pytest.skip(f"sample image missing: {path}")
    return path
