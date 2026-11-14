"""Tests for the circuit cleaner.

Most tests draw a small synthetic "circuit map" so each rule can be checked
directly and at different scales/colour schemes; a few run on the shipped
sample maps as real-data regressions.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from circuit_cleaner.clean_circuit import Settings, clean_circuit, clean_circuit_image, load_image

RAW = Settings(output_min_side=0)  # keep working size so masks line up
_SAMPLES = Path(__file__).resolve().parents[1] / "assets" / "samples"


def synthetic_map(scale: float = 1.0, dark: bool = False) -> np.ndarray:
    """A 900x600 map: black track loop, labels, corner ring, gray layout, red bits."""

    def p(*xy: float) -> tuple[int, ...]:
        return tuple(round(v * scale) for v in xy)

    h, w = p(600, 900)
    img = np.full((h, w, 3), 245, np.uint8)
    t = max(2, round(16 * scale))
    cv2.ellipse(img, p(450, 300), p(330, 200), 0, 0, 360, (190, 190, 190), t)  # gray layout
    cv2.ellipse(img, p(450, 300), p(300, 180), 0, 0, 360, (0, 0, 0), t)  # track
    cv2.putText(
        img,
        "Turn One",
        p(330, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2 * scale,
        (0, 0, 0),
        max(1, round(2 * scale)),
    )
    cv2.circle(
        img, p(780, 140), round(22 * scale), (0, 0, 0), max(1, round(2 * scale))
    )  # corner ring
    cv2.putText(
        img,
        "7",
        p(771, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * scale,
        (0, 0, 0),
        max(1, round(2 * scale)),
    )
    arrow = np.array([p(430, 98), p(470, 110), p(430, 122)], np.int32)  # red arrow next to track
    cv2.fillPoly(img, [arrow], (0, 0, 220))
    cv2.circle(
        img, p(450, 210), round(24 * scale), (0, 0, 220), max(1, round(3 * scale))
    )  # red ring
    cv2.fillPoly(
        img, [np.array([p(60, 40), p(90, 70), p(60, 100)], np.int32)], (0, 0, 220)
    )  # far red
    if dark:
        img = 255 - img
        img[(img[..., 2] < 60) & (img[..., 0] > 200)] = (0, 0, 220)  # keep the red red
    return img


def masks(out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, g, r = (out[..., i].astype(int) for i in range(3))
    red = (r > 120) & (g < 90) & (b < 90)
    black = (r + g + b < 3 * 110) & ~red
    return black, red


def at(mask: np.ndarray, scale: float, *xy: float) -> bool:
    x, y = (round(v * scale) for v in xy)
    return bool(mask[y - 2 : y + 3, x - 2 : x + 3].any())


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("dark", [False, True])
def test_synthetic_map_keeps_only_track_and_arrow(scale: float, dark: bool) -> None:
    img = synthetic_map(scale, dark)
    out = clean_circuit(img, RAW)
    k = out.shape[1] / img.shape[1]  # working-size factor
    black, red = masks(out)
    s = scale * k
    assert at(black, s, 150, 300) and at(black, s, 750, 300), "track missing"
    assert not at(black, s, 420, 300), "label text kept"
    assert not black[: round(170 * s), round(755 * s) : round(805 * s)].any(), "corner ring kept"
    assert not at(black, s, 120, 300), "gray layout kept"
    assert at(red, s, 445, 110), "red direction arrow dropped"
    assert not red[round(180 * s) : round(240 * s), round(420 * s) : round(480 * s)].any(), (
        "red ring kept"
    )
    assert not red[: round(110 * s), : round(110 * s)].any(), "far red shape kept"


def test_result_is_scale_invariant() -> None:
    """The same map at 0.5x and 2x gives (nearly) the same track."""
    small = clean_circuit(synthetic_map(0.5), RAW)
    big = clean_circuit(synthetic_map(2.0), RAW)
    big = cv2.resize(big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = masks(small)[0], masks(big)[0]
    assert (a & b).sum() / (a | b).sum() > 0.85


def test_track_with_corner_dots_stays_connected() -> None:
    """White corner dots cutting a thin track must not drop track sections."""
    img = np.full((500, 700, 3), 245, np.uint8)
    cv2.rectangle(img, (100, 100), (600, 400), (0, 0, 0), 6)
    for x, y in [(250, 100), (450, 100), (600, 250), (350, 400), (100, 250)]:
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
    black, _ = masks(clean_circuit(img, Settings(output_min_side=0, work_min_side=700)))
    for x, y in [
        (175, 100),
        (350, 100),
        (525, 100),
        (600, 175),
        (600, 325),
        (225, 400),
        (100, 175),
    ]:
        assert black[y - 4 : y + 5, x - 4 : x + 5].any(), f"section near ({x},{y}) lost"


def test_output_is_upscaled_to_min_side() -> None:
    out = clean_circuit(synthetic_map(0.5))
    assert max(out.shape[:2]) == Settings().output_min_side


def test_clean_circuit_image_writes_output(sample_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "a_cleaned.png"
    assert clean_circuit_image(sample_image, out) is True
    assert out.exists() and out.stat().st_size > 0


def test_clean_circuit_image_bad_path_returns_false(tmp_path: Path) -> None:
    assert clean_circuit_image(tmp_path / "missing.jpeg", tmp_path / "out.png") is False


def _red_pixels(path: Path) -> int:
    img = cv2.imread(str(path))
    assert img is not None
    return int(masks(img)[1].sum())


def test_red_corner_rings_removed_on_f_arrow_kept_on_a(tmp_path: Path) -> None:
    """Real data: f's red corner rings go, a's red start/finish arrow stays."""
    f_in, a_in = _SAMPLES / "f.jpeg", _SAMPLES / "a.jpeg"
    if not (f_in.exists() and a_in.exists()):
        pytest.skip("sample images missing")
    assert clean_circuit_image(f_in, tmp_path / "f.png")
    assert clean_circuit_image(a_in, tmp_path / "a.png")
    assert _red_pixels(tmp_path / "f.png") == 0
    assert _red_pixels(tmp_path / "a.png") > 0


@pytest.mark.parametrize("track_rgb", [(0, 0, 0), (255, 255, 255)])
def test_transparent_png_track_survives(tmp_path: Path, track_rgb: tuple[int, int, int]) -> None:
    """Black or white track on a fully transparent canvas is still found."""
    rgba = np.zeros((500, 700, 4), np.uint8)
    cv2.ellipse(rgba, (350, 250), (250, 150), 0, 0, 360, (*track_rgb, 255), 14)
    path = tmp_path / "transparent.png"
    Image.fromarray(rgba, "RGBA").save(path)
    img = load_image(path)
    assert img is not None
    black, _ = masks(clean_circuit(img, Settings(output_min_side=0, work_min_side=700)))
    for x, y in [(100, 250), (600, 250), (350, 100), (350, 400)]:
        assert black[y - 6 : y + 7, x - 6 : x + 7].any(), f"track lost near ({x},{y})"
    assert not black[240:260, 340:360].any(), "inside of the loop filled"


def test_coloured_canvas() -> None:
    """Black track on a saturated blue canvas with a lighter blue layout."""
    img = np.full((600, 900, 3), (160, 90, 30), np.uint8)  # BGR blue
    cv2.ellipse(img, (450, 300), (330, 200), 0, 0, 360, (200, 150, 110), 16)  # light layout
    cv2.ellipse(img, (450, 300), (300, 180), 0, 0, 360, (0, 0, 0), 16)
    black, _ = masks(clean_circuit(img, RAW))
    s = black.shape[1] / 900
    assert at(black, s, 150, 300) and at(black, s, 750, 300), "track missing"
    assert not at(black, s, 120, 300), "lighter layout kept"
