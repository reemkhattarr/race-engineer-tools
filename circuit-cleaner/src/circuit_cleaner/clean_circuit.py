"""Circuit map cleaner.

Turns a circuit-map image (track plus labels, corner numbers, gray secondary
layouts, logos, scale bars...) into a clean white image with only the track
in black and the red direction markers next to it.

How it works, in one pass and without fixed pixel constants:

1. Load any format (transparent PNG/WebP composited onto a contrasting
   backdrop) and measure each pixel's contrast with the background colour,
   so white, gray, tinted, coloured and dark canvases all become "dark ink
   on white".
2. Measure the track's stroke width ``W`` from the distance transform. Every
   size below is a multiple of ``W``, so the same rules work on a 450 px
   thumbnail and a 4000 px poster, with thin or thick tracks.
3. Ink = pixels clearly darker than the background (Otsu split, so lighter
   gray secondary layouts and watermarks fall out automatically).
4. A morphological opening with a disc of ~0.6 W erases everything drawn
   thinner than the track: text, corner-number rings, leader arrows, scale
   bars. What survives and is big is the track. Small gaps (e.g. white corner
   dots that cut a thin track) are bridged before choosing the big pieces.
5. Thin lines that hang off the track and are long and straight (pit lanes)
   are kept.
6. Red blobs are kept if they are solid (arrows, not rings) and close to the
   track, measured in multiples of ``W``.
7. The result is rendered using the source pixels' own darkness as opacity,
   so edges stay anti-aliased and the image can be upscaled without
   staircase edges.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

TRACK_BGR = (0, 0, 0)
RED_BGR = (0, 0, 210)


@dataclass(frozen=True)
class Settings:
    """Tunable knobs. Sizes are multiples of the measured track width W."""

    work_min_side: int = 1200  # upscale smaller images before processing
    work_max_side: int = 3000  # downscale larger ones (speed, no quality gain)
    output_min_side: int = 2000  # final image long side (0 = keep working size)
    opening: float = 0.6  # strokes thinner than this * W are not track
    bridge: float = 2.5  # gaps up to this * W are bridged when picking the track
    fill_dots: bool = True  # fill white corner dots/gaps drawn on the track
    keep_piece: float = 0.10  # keep track pieces >= this fraction of the largest
    keep_thin_lines: bool = True  # keep long straight thin lines attached to it
    thin_min_length: float = 6.0  # ... at least this * W long
    red_max_dist: float = 4.0  # red markers must be within this * W of the track
    red_min_solidity: float = 0.5  # arrows are solid, ring/arc markers are not
    red_max_area: float = 30.0  # in W^2; bigger red shapes are logos/compasses


DEFAULT = Settings()


def _disc(diameter: float) -> np.ndarray:
    d = max(1, round(diameter)) | 1  # odd size
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _resize(img: np.ndarray, s: float) -> np.ndarray:
    if abs(s - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def _keep_labels(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """uint8 mask of the label ids flagged in ``keep`` (index 0 = background)."""
    keep = keep.copy()
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def _reconstruct(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Connected components of ``mask`` that overlap ``seed``."""
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    hit = np.zeros(n, bool)
    hit[np.unique(labels[seed > 0]).astype(np.intp)] = True
    return _keep_labels(labels, hit)


def _darkness(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Gray image where ink is low and background is 255, whatever the scheme.

    Each pixel's darkness is its contrast with the background colour (median
    of the image border): lightness contrast normally, so a white track on
    black and a black track on beige both come out "dark ink on white", and
    lighter gray layouts stay light. If lightness alone can't see the track
    (a coloured track on an equally bright canvas) full Lab colour distance
    is used instead. Returns (lum, bg) with bg = 255.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0  # CIE L in 0..100 so distances are comparable
    b = max(2, min(img.shape[:2]) // 50)
    frame = (lab[:b], lab[-b:], lab[:, :b], lab[:, -b:])
    border = np.concatenate([f.reshape(-1, 3) for f in frame])
    diff = lab - np.median(border, axis=0)

    def reference(dist: np.ndarray) -> float:
        fg = dist[dist > 25.0]
        return float(np.percentile(fg, 90)) if fg.size else 0.0

    dist = np.abs(diff[..., 0])
    ref = reference(dist)
    if ref < 37.5:
        dist = np.linalg.norm(diff, axis=2)
        ref = reference(dist)
    ref = max(ref, 50.0)
    return 255.0 * (1.0 - np.clip(dist / ref, 0.0, 1.0)), 255.0


def _red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 10) | (h >= 170)) & (s >= 110) & (v >= 90)
    return red.astype(np.uint8) * 255


def _ink_mask(lum: np.ndarray, bg: float, red: np.ndarray) -> np.ndarray:
    """Pixels clearly darker than the background, split from lighter grays."""
    candidates = lum[(lum < bg - 20) & (red == 0)]
    if candidates.size < 50:
        thr = bg - 40
    else:
        thr, _ = cv2.threshold(
            candidates.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thr = float(np.clip(thr, 50, bg - 40))
    return ((lum < thr) & (red == 0)).astype(np.uint8) * 255


def _stroke_width(mask: np.ndarray) -> float:
    """Typical stroke width of the biggest ink shape (2 * median ridge distance)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 3.0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (dt >= 1) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (labels == biggest)
    vals = dt[ridge]
    return float(max(2.0, 2.0 * np.median(vals))) if vals.size else 3.0


def _track_region(ink: np.ndarray, w: float, cfg: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Return (core, region): thick strokes kept, and the bridged area they span."""
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _disc(cfg.opening * w))
    bridged = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _disc(cfg.bridge * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    if n <= 1:
        return core, bridged
    areas = stats[:, cv2.CC_STAT_AREA].astype(float)
    areas[0] = 0
    region = _keep_labels(labels, areas >= cfg.keep_piece * areas.max())
    return cv2.bitwise_and(core, region), region


def _thin_lines(ink: np.ndarray, track: np.ndarray, w: float, cfg: Settings) -> np.ndarray:
    """Long, straight, thin ink pieces touching the track (pit lanes, run-offs)."""
    rest = cv2.bitwise_and(ink, cv2.bitwise_not(track))
    near = cv2.dilate(track, _disc(3))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    dt = cv2.distanceTransform(rest, cv2.DIST_L2, 3)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if max(bw, bh) < cfg.thin_min_length * w:
            continue
        comp = labels[y : y + bh, x : x + bw] == i
        if not near[y : y + bh, x : x + bw][comp].any():
            continue
        thickness = max(1.0, 2.0 * float(np.median(dt[y : y + bh, x : x + bw][comp])))
        length = area / thickness
        pts = cv2.findNonZero(comp.astype(np.uint8))
        span = max(cv2.minAreaRect(pts)[1])
        # thin, long, and not curling back on itself like rings/arcs (span/length
        # of a circle is ~0.32; a pit-lane loop of two parallel lines is ~0.5)
        if thickness <= 0.7 * w and length >= cfg.thin_min_length * w and span >= 0.42 * length:
            keep[i] = True
    return _keep_labels(labels, keep)


def _red_markers(red: np.ndarray, track: np.ndarray, w: float, cfg: Settings) -> np.ndarray:
    # Judge shapes on slightly closed blobs so a ring broken into fragments
    # (JPEG, anti-aliasing) is still recognised as a ring, not as solid bits.
    grouped = cv2.morphologyEx(red, cv2.MORPH_CLOSE, _disc(0.4 * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    if n <= 1 or not track.any():
        return np.zeros_like(red)
    dist = cv2.distanceTransform(cv2.bitwise_not(track), cv2.DIST_L2, 3)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < (0.5 * w) ** 2 or area > cfg.red_max_area * w * w:
            continue
        comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
        if dist[y : y + bh, x : x + bw][comp > 0].min() > cfg.red_max_dist * w:
            continue
        hull = cv2.convexHull(cv2.findNonZero(comp))
        if area / max(cv2.contourArea(hull), 1.0) >= cfg.red_min_solidity:
            keep[i] = True
    return cv2.bitwise_and(red, _keep_labels(labels, keep))


def _dot_fill(ink: np.ndarray, region: np.ndarray, w: float) -> np.ndarray:
    """Small white gaps inside the track (corner dots), sized ~0.3-4 W^2.

    Checker-flag squares are smaller and long strips between parallel tracks
    are bigger, so both are left alone.
    """
    track = cv2.bitwise_and(ink, region)
    closed = cv2.morphologyEx(track, cv2.MORPH_CLOSE, _disc(1.5 * w))
    gaps = cv2.bitwise_and(closed, cv2.bitwise_not(ink))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (0.3 * w * w <= area <= 4 * w * w):
            continue
        # dots are roundish; the wedge inside a tight hairpin is long and thin
        (_, _), (rw, rh), _ = cv2.minAreaRect(
            cv2.findNonZero((labels[y : y + bh, x : x + bw] == i).astype(np.uint8))
        )
        if max(rw, rh) <= 2.0 * max(min(rw, rh), 1.0):
            keep[i] = True
    return _keep_labels(labels, keep)


def _opacity(lum: np.ndarray, bg: float, mask: np.ndarray, pad: int) -> np.ndarray:
    """Per-pixel opacity from the source's own darkness, limited to ``mask``.

    Using the source intensity keeps natural anti-aliased edges; the remap
    drops faint pixels (gray layouts, halos) that happen to sit in the mask.
    """
    roi = cv2.dilate(mask, _disc(2 * pad + 1)) > 0
    ink_level = np.percentile(lum[mask > 0], 5) if mask.any() else 0.0
    a = (bg - lum) / max(bg - ink_level, 1.0)
    a = np.clip((a - 0.3) / 0.5, 0.0, 1.0)
    return np.where(roi, a, 0.0).astype(np.float32)


def _redness(img: np.ndarray, mask: np.ndarray, pad: int) -> np.ndarray:
    """Opacity for red markers from how red each pixel is (works on any background)."""
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    roi = cv2.dilate(mask, _disc(2 * pad + 1)) > 0
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    a = np.clip((r - np.maximum(g, b) - 40.0) / 80.0, 0.0, 1.0)
    return np.where(roi, a, 0.0).astype(np.float32)


def clean_circuit(img_bgr: np.ndarray, cfg: Settings = DEFAULT) -> np.ndarray:
    """Clean a circuit map. BGR uint8 in, BGR uint8 out (white background)."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("expected a BGR colour image")
    long_side = max(img_bgr.shape[:2])
    scale = 1.0
    if long_side < cfg.work_min_side:
        scale = cfg.work_min_side / long_side
    elif long_side > cfg.work_max_side:
        scale = cfg.work_max_side / long_side
    img = _resize(img_bgr, scale)

    lum, bg = _darkness(img)
    red = _red_mask(img)
    ink = _ink_mask(lum, bg, red)
    w = _stroke_width(ink)

    core, region = _track_region(ink, w, cfg)
    # The region hugs the track footprint (incl. start/finish flag and corner
    # dots); rendering it from the source keeps those details as drawn. Also
    # take ink that is physically connected to the track and close to it:
    # pieces the opening thinned away (e.g. a track split by a drawn arrow).
    attached = _reconstruct(ink, core)
    keep = cv2.bitwise_or(region, cv2.bitwise_and(attached, cv2.dilate(core, _disc(2 * w))))
    track = cv2.bitwise_and(ink, keep)
    if cfg.keep_thin_lines:
        keep = cv2.bitwise_or(keep, _thin_lines(ink, track, w, cfg))
    markers = _red_markers(red, track, w, cfg)

    pad = max(1, round(0.15 * w))
    a_track = _opacity(lum, bg, keep, pad)
    if cfg.fill_dots:
        dots = cv2.GaussianBlur(_dot_fill(ink, region, w).astype(np.float32) / 255, (0, 0), 0.7)
        a_track = np.maximum(a_track, dots)
    a_red = _redness(img, markers, pad)
    out = np.full(img.shape, 255.0, np.float32)
    for alpha, colour in ((a_track, TRACK_BGR), (a_red, RED_BGR)):
        out = out * (1 - alpha[..., None]) + np.array(colour, np.float32) * alpha[..., None]
    out = out.astype(np.uint8)

    if cfg.output_min_side and max(out.shape[:2]) < cfg.output_min_side:
        s = cfg.output_min_side / max(out.shape[:2])
        out = cv2.resize(
            out,
            (round(out.shape[1] * s), round(out.shape[0] * s)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return out


def load_image(path: str | Path) -> np.ndarray | None:
    """Load any common raster file as 8-bit BGR, or None if it can't be read.

    Handles EXIF rotation, 16-bit / float / palette images, the first frame of
    animated GIF/WebP, and transparency: transparent pixels are composited
    onto white, or onto black when the visible content is itself light (a
    white track on a transparent canvas), so the track never vanishes.
    """
    try:
        with Image.open(path) as im:
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            if im.mode.startswith("I") or im.mode == "F":
                arr = np.asarray(im).astype(np.float64)
                hi = float(arr.max()) if arr.size else 1.0
                scale = 255.0 if hi <= 1.0 else 255.0 / (65535.0 if hi > 255 else 255.0)
                gray = np.clip(arr * scale, 0, 255).astype(np.uint8)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
                return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            rgba = np.asarray(im.convert("RGBA")).astype(np.float32)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    rgb, alpha = rgba[..., :3], rgba[..., 3:4] / 255.0
    visible = alpha[..., 0] > 0.5
    backdrop = 255.0
    if visible.any() and not visible.all():
        lum = rgb[visible] @ np.array([0.299, 0.587, 0.114], np.float32)
        backdrop = 0.0 if float(np.median(lum)) > 160 else 255.0
    out = np.clip(rgb * alpha + backdrop * (1.0 - alpha), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def clean_circuit_image(
    input_path: str | Path, output_path: str | Path, cfg: Settings = DEFAULT
) -> bool:
    """File wrapper around :func:`clean_circuit`. Returns False if unreadable.

    Inputs can be any format Pillow reads (incl. transparent PNG/WebP);
    outputs other than JPEG/PNG should be given a .png name by the caller.
    """
    img = load_image(input_path)
    if img is None:
        print(f"  ERROR: could not read {input_path}")
        return False
    out = clean_circuit(img, cfg)
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, 98]
        if str(output_path).lower().endswith((".jpg", ".jpeg"))
        else []
    )
    return bool(cv2.imwrite(str(output_path), out, params))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Clean circuit-map images (keeps track + red direction markers)."
    )
    p.add_argument("images", nargs="+", help="image files to clean")
    p.add_argument(
        "--no-thin-lines", action="store_true", help="drop pit lanes and other thin attached lines"
    )
    p.add_argument(
        "--size",
        type=int,
        default=DEFAULT.output_min_side,
        help="min output long side in px (0 = no upscale)",
    )
    args = p.parse_args(argv)
    cfg = Settings(keep_thin_lines=not args.no_thin_lines, output_min_side=args.size)
    ok = 0
    for path in args.images:
        src = Path(path)
        suffix = src.suffix if src.suffix.lower() in (".png", ".jpg", ".jpeg") else ".png"
        dst = src.with_name(f"{src.stem}_cleaned{suffix}")
        if clean_circuit_image(src, dst, cfg):
            ok += 1
            print(f"  {src.name} -> {dst.name}")
    print(f"Done. {ok}/{len(args.images)} cleaned.")
    return 0 if ok == len(args.images) else 1


if __name__ == "__main__":
    sys.exit(main())
