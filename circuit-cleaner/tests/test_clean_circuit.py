"""Smoke tests for the core cleaning function."""

from pathlib import Path

import cv2
import numpy as np

from circuit_cleaner.clean_circuit import clean_circuit_image, drop_arc_shaped_red


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


def _has_pixels(mask: np.ndarray) -> bool:
    return bool((mask > 0).sum() > 0)


def test_drop_arc_shaped_red_keeps_arrow_drops_arcs() -> None:
    """Solid direction arrows survive; thin corner-marker arcs/rings are dropped.

    This guards the fix for red corner markers (e.g. f.jpeg) without regressing
    the solid red start/finish arrows kept on the a-e samples.
    """
    arrow = np.zeros((400, 400), np.uint8)
    cv2.fillPoly(arrow, [np.array([[100, 100], [300, 200], [100, 300]])], 255)

    arc = np.zeros((400, 400), np.uint8)
    cv2.ellipse(arc, (200, 200), (120, 120), 0, 40, 320, 255, thickness=14)

    ring = np.zeros((400, 400), np.uint8)
    cv2.circle(ring, (200, 200), 120, 255, thickness=14)

    assert _has_pixels(drop_arc_shaped_red(arrow))  # solid arrow kept
    assert not _has_pixels(drop_arc_shaped_red(arc))  # open arc dropped
    assert not _has_pixels(drop_arc_shaped_red(ring))  # ring dropped
