"""Smoke tests for the core cleaning function."""

from pathlib import Path

from circuit_cleaner.clean_circuit import clean_circuit_image


def test_clean_circuit_image_writes_output(sample_image: Path, tmp_path: Path) -> None:
    """Cleaning a known-good sample returns True and writes a non-empty file."""
    out = tmp_path / "a_cleaned.jpeg"

    ok = clean_circuit_image(str(sample_image), str(out), add_numbers=False)

    assert ok is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_clean_circuit_image_bad_path_returns_false(tmp_path: Path) -> None:
    """A missing/unreadable input returns False rather than raising."""
    out = tmp_path / "out.jpeg"

    ok = clean_circuit_image(str(tmp_path / "does_not_exist.jpeg"), str(out))

    assert ok is False
