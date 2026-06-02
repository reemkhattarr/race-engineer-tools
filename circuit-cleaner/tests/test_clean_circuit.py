"""Smoke tests for the core cleaning function."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from circuit_cleaner.clean_circuit import (
    clean_circuit_image,
    drop_arc_shaped_red,
    filter_red_near_circuit,
)


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


def test_red_proximity_scales_to_keep_large_markers() -> None:
    """A wider proximity keeps a marker that a too-tight radius would clip.

    Guards the upscale-clipping fix: RED_PROXIMITY is in original-image pixels,
    so on an upscaled image the radius must scale up or far parts of a big
    marker (e.g. a direction arrow) get cut off.
    """
    dark = np.zeros((200, 200), np.uint8)
    dark[10, :] = 255  # a horizontal "track" line near the top
    red = np.zeros((200, 200), np.uint8)
    red[120:160, 80:120] = 255  # a solid marker ~110-150 px below the line

    tight = filter_red_near_circuit(red, dark, proximity=30)
    wide = filter_red_near_circuit(red, dark, proximity=180)

    assert (tight > 0).sum() == 0  # marker clipped entirely at a tight radius
    assert (wide > 0).sum() == int((red > 0).sum())  # fully kept when scaled up


def _red_pixel_count(image_path: Path) -> int:
    img = cv2.imread(str(image_path))
    assert img is not None
    r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    return int(((r > 150) & (g < 90) & (b < 90)).sum())


_SAMPLES = Path(__file__).resolve().parents[1] / "assets" / "samples"


def test_red_corner_rings_removed_on_f(tmp_path: Path) -> None:
    """f.jpeg's red corner rings are gone, while a.jpeg keeps its red arrow.

    Real-data regression for both directions of the shape filter: arcs/rings
    dropped (f), solid direction arrows preserved (a).
    """
    f_in = _SAMPLES / "f.jpeg"
    a_in = _SAMPLES / "a.jpeg"
    if not (f_in.exists() and a_in.exists()):
        pytest.skip("sample images missing")

    f_out = tmp_path / "f_cleaned.png"
    a_out = tmp_path / "a_cleaned.png"
    assert clean_circuit_image(str(f_in), str(f_out), add_numbers=False)
    assert clean_circuit_image(str(a_in), str(a_out), add_numbers=False)

    assert _red_pixel_count(f_out) == 0  # all red corner rings removed
    assert _red_pixel_count(a_out) > 0  # red start/finish arrow preserved
