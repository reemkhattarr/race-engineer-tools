#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# This file is the original, working clean_circuit tool, moved into the
# race-engineer-tools suite verbatim except for the lazy pytesseract import
# below. It is excluded from ruff format/lint (see root pyproject.toml) and the
# above pyright suppressions cover the optional add-numbers/Gemini deps that the
# lean install omits — so the proven cleaning logic stays byte-stable. New code
# lives in serving/ and is fully linted and type-checked.
"""
Circuit map image cleaner.

Transforms circuit layout images (e.g. a.jpeg → a_cleaned.jpeg) by:
  - Removing all text, labels and external objects
  - Removing gray secondary-track regions
  - Replacing circled corner numbers with plain red numbers
  - Keeping red direction markers (chevrons/arrows) near the circuit
  - Producing a clean white-background circuit outline

Usage:
    python3 clean_circuit.py                          # Hough+OCR, all images in cwd
    python3 clean_circuit.py a.jpeg b.jpeg            # specific files
    python3 clean_circuit.py --gemini-key KEY *.jpeg  # Gemini vision (recommended)
    GEMINI_API_KEY=KEY python3 clean_circuit.py       # key from environment

Get a free Gemini key at https://aistudio.google.com/app/apikey
Install the client once:  pip install google-genai
"""

import os
import sys
import glob
import json
import base64
import argparse
from collections import Counter

# Load .env if present (picks up GEMINI_API_KEY without requiring shell export)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# pytesseract is only used by the (ignored) --add-numbers corner-numbering path.
# Import it lazily so the module loads on a lean install (e.g. the HF Space)
# where Tesseract is not present. The default add_numbers=False path never
# touches it.
try:
    import pytesseract

    import shutil as _shutil
    pytesseract.pytesseract.tesseract_cmd = (
        _shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    )
except ImportError:
    pytesseract = None

# ── Parameters ───────────────────────────────────────────────────────────────
DARK_THRESHOLD     = 80
DARK_THRESHOLD_MAX = 140
RED_MIN_R          = 140
RED_MAX_GB         = 90
RED_PROXIMITY      = 60      # px — red pixels further than this from circuit are removed
RED_MIN_SOLIDITY   = 0.50    # solidity gap is wide: arrows score ~0.66-1.0, corner arcs/rings ~0.13-0.45; sit in the middle to keep arrows safely while dropping arcs
RED_SHAPE_MIN_AREA = 60      # px — ignore tiny red specks when judging shape (noise, not markers)
CC_KEEP_FRAC       = 0.015   # min area as fraction of largest dark component
CIRCUIT_BB_PAD     = 0.08    # fractional padding for off-circuit title removal
HOUGH_PARAM2       = 18      # accumulator threshold — lower = more detections
HOUGH_MIN_R        = 8       # base; scaled with image dim by _hough_radii()
HOUGH_MAX_R        = 50      # base; scaled with image dim by _hough_radii()
MAX_CORNER_NUM     = 25
ON_CIRCUIT_MIN     = 3       # ring boundary must touch this many circuit pixels
FONT_SCALE         = 0.85    # red number size relative to circle diameter
# ─────────────────────────────────────────────────────────────────────────────


# ── Preprocessing ─────────────────────────────────────────────────────────────

MIN_LONG_SIDE = 4000   # upscale so the longest side is at least this many pixels


def upscale_if_needed(img_bgr):
    """
    Upscale small images so processing happens at a resolution where circuit
    lines are smooth curves rather than staircase pixels.  Images already at or
    above MIN_LONG_SIDE are returned unchanged.
    LANCZOS4 gives the cleanest sub-pixel interpolation for line art.
    """
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side >= MIN_LONG_SIDE:
        return img_bgr, 1.0
    scale  = MIN_LONG_SIDE / long_side
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))
    up     = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    print(f"    Upscaled {w}×{h} → {new_w}×{new_h} (×{scale:.2f})")
    return up, scale


def sharpen_image(img_bgr, strength=1.5, sigma=1.0):
    """
    Unsharp mask applied after upscaling.  A lower strength and tighter sigma
    are used here because the image is already large; we just want to recover
    a little anti-aliasing softness introduced by LANCZOS4, not amplify noise.
    """
    blurred   = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img_bgr, strength, blurred, -(strength - 1), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_background(img_bgr):
    """
    If the image has a non-white background (e.g. the light gray in d.jpeg),
    normalise it to white by sampling border pixels and rescaling uniformly.
    Images that are already near-white are returned unchanged.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border = np.concatenate([
        gray[:12, :].ravel(), gray[-12:, :].ravel(),
        gray[:, :12].ravel(), gray[:, -12:].ravel(),
    ])
    bg = int(np.median(border))
    if bg >= 240:
        return img_bgr
    out = img_bgr.astype(np.float32) * (255.0 / max(bg, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def _hough_radii(h, w):
    """
    Scale ring-radius search bounds with image size.  Corner-marker rings on
    circuit maps sit at roughly 0.7-2.5% of the shorter image side; the defaults
    were calibrated for ~1280 px images and fall short on upscaled 4000+ px ones.
    """
    short = min(h, w)
    r_min = max(HOUGH_MIN_R, int(round(short * 0.006)))
    r_max = max(HOUGH_MAX_R, int(round(short * 0.035)))
    return r_min, r_max


# ── Colour masks ──────────────────────────────────────────────────────────────

def auto_dark_threshold(img_bgr):
    """
    Select the lowest threshold in [DARK_THRESHOLD, DARK_THRESHOLD_MAX] where
    the largest dark connected component dominates (ratio to 2nd >= 50).
    This handles thin-line images (e.g. b.jpeg) that need a higher threshold
    to fuse anti-aliased edges into one connected region.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for t in range(DARK_THRESHOLD, DARK_THRESHOLD_MAX + 1, 10):
        dark = (gray < t).astype(np.uint8) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n <= 2:
            continue
        areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
        if areas[0] / max(areas[1], 1) >= 50:
            return t
    return DARK_THRESHOLD_MAX


def make_dark_mask(img_bgr, threshold=None):
    """Binary mask of all dark (circuit) pixels."""
    if threshold is None:
        threshold = auto_dark_threshold(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return (gray < threshold).astype(np.uint8) * 255, threshold


def make_red_mask(img_bgr):
    """Binary mask of red marker pixels (chevrons, direction arrows)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = (r > RED_MIN_R) & (g < RED_MAX_GB) & (b < RED_MAX_GB)
    return red.astype(np.uint8) * 255


# ── Circuit-component cleanup ─────────────────────────────────────────────────

def keep_circuit_components(dark_mask):
    """
    Drop dark connected components whose area is below CC_KEEP_FRAC of the
    largest component, or whose centroid falls outside the padded bounding box
    of the largest component.  This removes text labels, gray-track outlines,
    compass glyphs, scale bars, and off-track titles.
    """
    h, w = dark_mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8
    )
    if n <= 1:
        return dark_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area  = int(np.max(areas))
    min_area      = max(200, int(largest_area * CC_KEEP_FRAC))
    largest_label = int(np.argmax(areas)) + 1

    lx = stats[largest_label, cv2.CC_STAT_LEFT]
    ly = stats[largest_label, cv2.CC_STAT_TOP]
    lw = stats[largest_label, cv2.CC_STAT_WIDTH]
    lh = stats[largest_label, cv2.CC_STAT_HEIGHT]
    px, py = int(w * CIRCUIT_BB_PAD), int(h * CIRCUIT_BB_PAD)
    bb_x1, bb_y1 = max(0, lx - px),      max(0, ly - py)
    bb_x2, bb_y2 = min(w, lx + lw + px), min(h, ly + lh + py)

    clean = np.zeros_like(dark_mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[i]
        if not (bb_x1 <= cx <= bb_x2 and bb_y1 <= cy <= bb_y2):
            continue
        clean[labels == i] = 255
    return clean


def restore_near_track_dark_details(raw_dark, dark_clean):
    """
    Recover small dark details that sit immediately next to the track.

    This preserves legitimate map markers like start/finish checker blocks that
    can be dropped by the coarse connected-component area filter, while still
    rejecting distant text/noise components.
    """
    # Candidates are dark pixels present in the original threshold mask but
    # removed by the component cleanup stage.
    removed = cv2.bitwise_and(raw_dark, cv2.bitwise_not(dark_clean))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(removed, connectivity=8)
    if n <= 1:
        return dark_clean

    # Only keep details that are close to already accepted track pixels.
    near_track = cv2.dilate(
        dark_clean,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # ~20 px halo
    )

    restored = dark_clean.copy()
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Skip tiny specks and large blobs (likely labels/graphics).
        if area < 18 or area > 3500:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        if np.any(cv2.bitwise_and(comp, near_track)):
            restored[comp > 0] = 255
    return restored


def filter_red_near_circuit(red_mask, dark_mask):
    """Keep only red pixels within RED_PROXIMITY px of a circuit line."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (RED_PROXIMITY * 2 + 1, RED_PROXIMITY * 2 + 1)
    )
    return cv2.bitwise_and(red_mask, cv2.dilate(dark_mask, kernel))


def drop_arc_shaped_red(red_mask):
    """Drop thin open-arc / ring red blobs (corner markers); keep solid arrows.

    Direction markers (arrowheads / chevrons) are near-solid: their pixel area
    nearly fills their convex hull, so solidity = area / hull_area is high
    (~0.9-1.0 on the sample circuits). Corner-number markers drawn as thin red
    arcs or rings have low solidity — the convex hull is a full disk while the
    stroke is only a sliver. We keep high-solidity blobs (arrows) and drop the
    low-solidity ones (corner markers). Tiny specks below RED_SHAPE_MIN_AREA are
    left untouched (too small to classify; proximity already vetted them).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (red_mask > 0).astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(red_mask)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = (labels == i).astype(np.uint8)
        if area < RED_SHAPE_MIN_AREA:
            kept[comp > 0] = 255
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = max(cv2.contourArea(cv2.convexHull(np.vstack(contours))), 1.0)
        if area / hull_area >= RED_MIN_SOLIDITY:
            kept[comp > 0] = 255
    return kept


# ── Circle OCR ────────────────────────────────────────────────────────────────

def _ocr_circle(gray, x, y, r, w, h):
    """
    OCR a single circle crop.  Returns (number_str, confidence) or (None, 0).

    Strategy:
      - Crop the circle interior, upscale to ~96 px
      - Apply CLAHE for local contrast normalisation
      - Generate 8 image variants (Otsu / adaptive, normal / inverted, dilated)
      - Run Tesseract with PSM 6, 7, 8, 13 on each variant
      - Majority vote; flag 6/9 ambiguity when vote split is tight
    """
    pad = max(2, r // 5)
    x1, y1 = max(0, x - r + pad), max(0, y - r + pad)
    x2, y2 = min(w, x + r - pad), min(h, y + r - pad)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    roi = gray[y1:y2, x1:x2]
    # Scale to at least 128px on the longest side; minimum 6x for tiny circles
    target = 128
    side   = max(roi.shape[0], roi.shape[1], 1)
    scale  = max(6, target // side)
    roi_up = cv2.resize(roi, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_LANCZOS4)

    # Mask out the circle ring — only the interior (digit area) should reach OCR.
    # The ring is at radius ~r-pad; we keep a circular interior of radius r-pad-3.
    uh, uw = roi_up.shape[:2]
    interior_r = max(1, int((r - pad - 3) * scale))
    ring_mask  = np.zeros((uh, uw), np.uint8)
    cv2.circle(ring_mask, (uw // 2, uh // 2), interior_r, 255, -1)
    roi_masked = np.where(ring_mask > 0, roi_up, 255).astype(np.uint8)

    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    roi_enh = clahe.apply(roi_masked)

    _, roi_otsu = cv2.threshold(
        roi_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    roi_adapt = cv2.adaptiveThreshold(
        roi_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Fatten thin strokes: eroding the binary image expands the dark (digit) pixels
    k3 = np.ones((3, 3), np.uint8)
    roi_otsu_fat = cv2.erode(roi_otsu, k3)

    roi_adapt = cv2.adaptiveThreshold(
        roi_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 4 variants × 2 PSM = 8 OCR calls per circle
    # fattened otsu helps thin digits (e.g. "1" in small circles)
    # adaptive threshold handles uneven illumination
    variants = [
        roi_otsu,
        roi_otsu_fat,
        roi_adapt,
        cv2.bitwise_not(roi_adapt),
    ]

    votes = []
    cfg = "--oem 3 -c tessedit_char_whitelist=0123456789"
    for img_v in variants:
        for psm in [8, 7]:
            try:
                txt = pytesseract.image_to_string(
                    img_v, config=f"--psm {psm} {cfg}"
                ).strip()
                if txt.isdigit() and 1 <= int(txt) <= MAX_CORNER_NUM:
                    votes.append(txt)
            except Exception:
                pass

    if not votes:
        return None, 0

    top  = Counter(votes).most_common(2)
    best = top[0][0]
    conf = top[0][1] / len(votes)

    # Flag low-confidence 6/9 ambiguity so global resolution can handle it
    if (len(top) > 1
            and top[0][0] in ('6', '9')
            and top[1][0] in ('6', '9')
            and top[0][1] - top[1][1] <= 2):
        conf = 0.3

    return best, conf


# ── Circle detection ──────────────────────────────────────────────────────────

def detect_circled_numbers(img_bgr, circuit_mask):
    """
    Detect circled corner numbers using Hough circles + per-crop OCR.

    Returns:
      numbered  – [(x, y, r, num), ...] deduplicated per unique number
      all_valid – [(x, y, r), ...]      every geometrically valid circle,
                                         used to erase rings even when OCR failed
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Scale accumulator threshold with image size: larger images have more
    # curvature in the track and generate more false-positive circle detections.
    img_area = h * w
    if img_area > 1_200_000:
        p2 = HOUGH_PARAM2 + 8
    elif img_area > 600_000:
        p2 = HOUGH_PARAM2 + 4
    else:
        p2 = HOUGH_PARAM2

    raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=18,
        param1=50, param2=p2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    if raw is None:
        return [], []

    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    candidates = []
    all_valid   = []

    for x, y, r in np.round(raw[0]).astype(int):


        # ── On-circuit check: ring boundary must be adjacent to circuit line ──
        on_circuit = 0
        for check_r in [r - 2, r, r + 3, r + 6, r + 10, r + 15]:
            if check_r < 1:
                continue
            bx = np.clip((x + check_r * cos_a).astype(int), 0, w - 1)
            by = np.clip((y + check_r * sin_a).astype(int), 0, h - 1)
            on_circuit = max(on_circuit, int(np.sum(circuit_mask[by, bx] > 0)))
        if on_circuit < ON_CIRCUIT_MIN:
            continue

        # ── Interior brightness and digit-presence checks ─────────────────────
        interior = np.zeros((h, w), np.uint8)
        cv2.circle(interior, (x, y), max(1, r - 4), 255, -1)
        ipx = gray[interior > 0]
        if len(ipx) == 0 or np.mean(ipx) < 130:
            continue
        if np.sum(ipx < 100) < 2:
            continue

        # all_valid collects every geometrically sound circle for ring erasure,
        # even those with a high dark fraction (circuit line through interior).
        dark_frac = float(np.sum(ipx < 100)) / len(ipx)
        all_valid.append((x, y, r))

        # For OCR candidates, record dark_frac as a tie-breaker: when two circles
        # read the same number at equal confidence, prefer the one with less dark
        # interior (cleaner white background = more likely a real digit circle).
        num, conf = _ocr_circle(gray, x, y, r, w, h)
        if num is not None:
            candidates.append((x, y, r, num, conf, on_circuit, dark_frac))

    # ── Deduplicate all_valid geometrically (Euclidean 15 px) ────────────────
    all_valid_dd = []
    for x, y, r in all_valid:
        if not any((x - ex) ** 2 + (y - ey) ** 2 < 225
                   for ex, ey, _ in all_valid_dd):
            all_valid_dd.append((x, y, r))

    # ── Deduplicate candidates: merge near-duplicate detections ──────────────
    # Sort by confidence descending so the best reading wins each cluster.
    # Euclidean 15 px threshold: prevents two genuinely distinct circles from
    # being merged (the old L∞ < 15 check silently merged circles up to ~21 px
    # apart whenever both axis differences happened to be < 15).
    cands_dd = []
    for item in sorted(candidates, key=lambda c: -c[4]):
        x, y = item[0], item[1]
        if not any((x - ex) ** 2 + (y - ey) ** 2 < 225
                   for ex, ey, *_ in cands_dd):
            cands_dd.append(item)

    # ── Keep best reading per number ──────────────────────────────────────────
    # Primary sort key: OCR confidence (higher is better).
    # Tie-breaker: interior dark fraction (lower = cleaner digit background).
    # A track-curve false positive that coincidentally reads the same digit as a
    # real corner tends to have a higher dark fraction (circuit line through the
    # interior inflates the dark-pixel count).
    by_num: dict[str, tuple] = {}
    for entry in cands_dd:
        num, conf, df = entry[3], entry[4], entry[6]
        if num not in by_num:
            by_num[num] = entry
        else:
            prev_conf, prev_df = by_num[num][4], by_num[num][6]
            if conf > prev_conf or (conf == prev_conf and df < prev_df):
                by_num[num] = entry

    numbered = [(x, y, r, num) for x, y, r, num, *_ in by_num.values()]

    # ── Drop single-digit shadows near two-digit twins ────────────────────────
    # e.g. if "1" is detected close to "11" or "10", it is a mis-crop of those.
    positions = {num: (x, y) for x, y, r, num in numbered}
    filtered = []
    for x, y, r, num in numbered:
        if len(num) == 1:
            shadow = any(
                len(n2) == 2 and n2[0] == num
                and ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5 < r * 4
                for n2, (x2, y2) in positions.items()
            )
            if shadow:
                continue
        filtered.append((x, y, r, num))

    return filtered, all_valid_dd


# ── Loose ring scanner (for erasure only, no OCR / on-circuit check) ─────────

def find_all_rings(img_bgr):
    """
    Run a permissive Hough circle scan to find every ring-shaped annotation
    in the image regardless of whether it touches the circuit.  Results are
    used exclusively for ring + arrow erasure — not for corner numbering.

    Criteria (no on-circuit check):
      - Interior mean brightness >= 140  (rings have a bright/white inside)
      - At least 1 dark interior pixel   (there must be *some* digit/mark)
      - Deduplicated at 15 px Euclidean distance
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=15,
        param1=50, param2=20,  # lower threshold -> fewer missed corner rings
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    if raw is None:
        return []

    angles  = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cos_a   = np.cos(angles)
    sin_a   = np.sin(angles)

    found = []
    for x, y, r in np.round(raw[0]).astype(int):
        # Interior must be mostly bright (white inside the ring)
        interior = np.zeros((h, w), np.uint8)
        cv2.circle(interior, (x, y), max(1, r - 4), 255, -1)
        ipx = gray[interior > 0]
        if len(ipx) == 0 or np.mean(ipx) < 150:
            continue
        if np.sum(ipx < 100) < 2:
            continue

        # Ring arc must actually be dark: sample pixels on the perimeter
        bx = np.clip((x + r * cos_a).astype(int), 0, w - 1)
        by = np.clip((y + r * sin_a).astype(int), 0, h - 1)
        arc_dark_frac = float(np.sum(gray[by, bx] < 100)) / len(angles)
        if arc_dark_frac < 0.22:   # allow thinner/aliased rings
            continue

        # Deduplicate at 15 px
        if not any((x - ex) ** 2 + (y - ey) ** 2 < 225 for ex, ey, _ in found):
            found.append((x, y, r))

    return found


def extract_digit_pixels_in_circle(gray_img, x, y, r, digit_thr):
    """
    Return a mask of likely digit strokes inside a detected annotation ring.

    We keep dark components fully enclosed by the interior mask, and discard
    components touching the interior boundary (usually track strokes crossing the
    circle or residual ring pixels).
    """
    h, w = gray_img.shape
    interior_r = max(3, r - 6)
    interior = np.zeros((h, w), np.uint8)
    cv2.circle(interior, (x, y), interior_r, 255, -1)

    dark_inside = ((gray_img < digit_thr) & (interior > 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_inside, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8)

    edge_band = cv2.bitwise_xor(
        interior,
        cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    )
    digit_mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        edge_touch = cv2.bitwise_and(comp, edge_band)
        edge_ratio = float(np.sum(edge_touch > 0)) / max(np.sum(comp > 0), 1)
        # Keep mostly enclosed components even if a few pixels touch the edge.
        if edge_ratio > 0.60:
            continue
        digit_mask[comp > 0] = 255
    # Fallback: if strict filtering found nothing, keep interior dark pixels but
    # trim 3 px from boundary to avoid reintroducing the ring stroke.
    if np.sum(digit_mask > 0) < 2:
        safe_interior = cv2.erode(
            interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        fallback = ((gray_img < digit_thr) & (safe_interior > 0)).astype(np.uint8) * 255
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(fallback, connectivity=8)
        for j in range(1, n2):
            area = int(stats2[j, cv2.CC_STAT_AREA])
            if 2 <= area <= max(600, r * r):
                digit_mask[labels2 == j] = 255
    return digit_mask


def keep_rings_near_track(rings, track_mask, halo_px=36, min_overlap=8):
    """Keep only rings whose boundary lies near the cleaned track."""
    if not rings:
        return []
    h, w = track_mask.shape
    near_track = cv2.dilate(
        track_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_px + 1, 2 * halo_px + 1)),
    )
    kept = []
    for x, y, r in rings:
        ring = np.zeros((h, w), np.uint8)
        cv2.circle(ring, (x, y), max(1, r + 2), 255, 2)
        overlap = int(np.sum((ring > 0) & (near_track > 0)))
        if overlap >= min_overlap:
            kept.append((x, y, r))
    return kept


# ── Gemini-based corner detection ────────────────────────────────────────────

def find_ring_radius(gray, x, y, w, h):
    """
    Scan concentric rings around (x, y) to find the radius where dark pixels
    are most concentrated — this is the actual ring radius.  Hough tends to
    underestimate by ~8 px; Gemini gives us the centre but no radius at all.
    """
    angles  = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    cos_a   = np.cos(angles)
    sin_a   = np.sin(angles)
    r_min, r_max = _hough_radii(h, w)
    best_r, best_cnt = r_min, 0
    for r in range(r_min, r_max + 1):
        bx  = np.clip((x + r * cos_a).astype(int), 0, w - 1)
        by  = np.clip((y + r * sin_a).astype(int), 0, h - 1)
        cnt = int(np.sum(gray[by, bx] < 80))
        if cnt > best_cnt:
            best_cnt = cnt
            best_r   = r
    return best_r


def find_ring_band(gray, x, y, w, h, r_seed):
    """
    Locate the exact ring stroke band around (x, y) by sweeping radii and
    keeping every radius where the perimeter is predominantly dark.

    Returns (r_inner, r_peak, r_outer).  The outer ring stroke goes from
    r_inner to r_outer; the digit lives strictly inside r_inner.
    """
    angles  = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    r_lo = max(HOUGH_MIN_R - 2, 3)
    r_hi = min(HOUGH_MAX_R + 14, min(w, h) // 2 - 1)
    counts = np.zeros(r_hi - r_lo + 1, dtype=np.int32)
    for i, r in enumerate(range(r_lo, r_hi + 1)):
        bx = np.clip((x + r * cos_a).astype(int), 0, w - 1)
        by = np.clip((y + r * sin_a).astype(int), 0, h - 1)
        counts[i] = int(np.sum(gray[by, bx] < 100))

    # Restrict the search to radii near the seed so a nearby track curve
    # doesn't pull the peak away from the ring.
    search_lo = max(0, r_seed - r_lo - 8)
    search_hi = min(len(counts) - 1, r_seed - r_lo + 8)
    if search_hi <= search_lo:
        return r_seed - 2, r_seed, r_seed + 2

    local = counts[search_lo:search_hi + 1]
    peak_idx_local = int(np.argmax(local))
    peak_idx = search_lo + peak_idx_local
    peak_count = int(local[peak_idx_local])
    if peak_count < 8:
        # Ring is faint — be conservative
        return r_seed - 3, r_seed, r_seed + 3

    thr = max(peak_count // 2, 6)
    # Walk down from peak until counts fall below threshold
    lo = peak_idx
    while lo > 0 and counts[lo - 1] >= thr:
        lo -= 1
    hi = peak_idx
    while hi < len(counts) - 1 and counts[hi + 1] >= thr:
        hi += 1

    r_peak = r_lo + peak_idx
    r_inner = max(r_lo, r_lo + lo)
    r_outer = r_lo + hi
    # Inflate slightly to swallow anti-aliased edge pixels
    return max(2, r_inner - 1), r_peak, r_outer + 1


def detect_corners_gemini(img_bgr, api_key):
    """
    Use Gemini 2.0 Flash vision to locate corner-marker circles.

    Returns:
      circles    – [(x, y, r, num_str), ...] one entry per corner number
      all_circles – [(x, y, r), ...]         same entries (for ring erasure)

    The function is a drop-in replacement for detect_circled_numbers: the rest
    of the pipeline (ring erasure, number drawing) is completely unchanged.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("  ERROR: google-genai not installed.")
        print("         Run:  pip install google-genai")
        return [], []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Resize to a fixed working size so Gemini-returned coords map back
    # deterministically.  Very small images get upscaled (so circles are big
    # enough to see); very large images get downscaled (so Gemini's internal
    # vision pipeline doesn't silently rescale coords on us).
    target_max = 1400
    target_min = 1200
    send_bgr = img_bgr
    scale    = 1.0
    if max(h, w) > target_max:
        scale    = target_max / max(h, w)
        new_w    = int(round(w * scale))
        new_h    = int(round(h * scale))
        send_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"    Downscaled {w}x{h} → {new_w}x{new_h} for Gemini")
    elif min(h, w) < target_min:
        scale    = target_min / min(h, w)
        new_w    = int(round(w * scale))
        new_h    = int(round(h * scale))
        send_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"    Upscaled {w}x{h} → {new_w}x{new_h} for Gemini")

    # Encode image as JPEG bytes for the Gemini inline_data part
    ok, buf = cv2.imencode(".jpg", send_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        print("  ERROR: could not encode image for Gemini")
        return [], []
    img_bytes = buf.tobytes()

    prompt = (
        "This is a racing circuit map diagram. "
        "Corner markers are small BLACK CIRCLES (rings) each containing a NUMBER inside. "
        "They mark every numbered corner of the racing circuit, starting from 1. "
        "Look carefully across the ENTIRE image — corners are often tightly packed or near track lines. "
        "Return ONLY a JSON object: keys are corner numbers as strings, values are [y, x] "
        "coordinates of each circle centre, NORMALIZED to the range 0-1000 "
        "(y=0 is the top edge, y=1000 is the bottom edge; x=0 is the left edge, x=1000 is the right edge). "
        "Include every visible corner number without exception. "
        "No explanation, no markdown, just the raw JSON object."
    )

    client   = genai.Client(api_key=api_key)
    img_part = genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    def parse_gemini_json(text):
        """Parse a Gemini response into {num_str: (cx_scaled, cy_scaled)} dict."""
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(l for l in t.splitlines() if not l.startswith("```")).strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return {}

    def call_gemini(contents):
        import re, time
        last_exc = None
        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=model_name, contents=contents)
                    print(f"    Gemini model used: {model_name}")
                    return resp.text
                except Exception as exc:
                    msg = str(exc)
                    # Invalid key — no point retrying any model
                    if "400" in msg and ("INVALID_ARGUMENT" in msg or "API_KEY_INVALID" in msg):
                        raise exc
                    # Quota fully exhausted (daily limit) — no point retrying
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        # If error message contains a short retry delay it's per-minute
                        # rate limiting (worth a short wait).  If not, it's daily quota
                        # exhaustion — fail immediately so we don't hang for minutes.
                        m = re.search(r"retry[_ ]in ([\d.]+)s", msg, re.I)
                        if m:
                            wait = min(float(m.group(1)) + 2, 30)
                            print(f"    Rate limited on {model_name}, waiting {wait:.0f}s...")
                            time.sleep(wait)
                            last_exc = exc
                            continue   # retry same model after short wait
                        # No suggested delay → daily quota exhausted, skip to next model fast
                        print(f"    Quota exhausted on {model_name}, trying next model...")
                        last_exc = exc
                        break
                    last_exc = exc
                    break          # other error, try next model
        raise last_exc

    # ── First pass ────────────────────────────────────────────────────────────
    try:
        raw_text = call_gemini([img_part, prompt])
    except Exception as exc:
        print(f"  ERROR: Gemini API call failed: {exc}")
        return [], []

    data = parse_gemini_json(raw_text)
    if not data:
        print(f"  ERROR: Gemini returned non-JSON: {raw_text[:300]}")
        return [], []

    found_nums = {str(k) for k in data if str(k).isdigit()}

    # ── Second pass: ask for any gaps in the number sequence ─────────────────
    if found_nums:
        max_found = max(int(n) for n in found_nums)
        expected  = {str(i) for i in range(1, max_found + 1)}
        missing   = sorted(expected - found_nums, key=int)
        if missing:
            print(f"    First pass missing: {missing} — running focused second pass")
            missing_str = ", ".join(missing)
            prompt2 = (
                f"Look again at this racing circuit map. "
                f"In the first pass these corner numbers were NOT found: {missing_str}. "
                f"Search carefully for black circle markers containing those specific numbers. "
                f"They may be small, partly overlapping with track lines, or near other corners. "
                f"Return ONLY a JSON object with the missing corners you can now find: "
                f"keys are corner numbers as strings, values are [x, y] pixel coordinates. "
                f"If you genuinely cannot find a corner, omit it. "
                f"No explanation, no markdown, just raw JSON."
            )
            try:
                raw2 = call_gemini([img_part, prompt2])
                data2 = parse_gemini_json(raw2)
                # Merge: second pass fills gaps only (don't overwrite first-pass results)
                for k, v in data2.items():
                    if str(k) not in data:
                        data[str(k)] = v
            except Exception as exc:
                print(f"    Second pass failed (non-fatal): {exc}")

    # ── Build result lists ────────────────────────────────────────────────────
    # We requested coords in [y, x] normalized 0-1000 form (Gemini's preferred
    # spatial format), and map them back to the ORIGINAL input dimensions (w, h).
    circles     = []
    all_circles = []
    for num_str, coords in data.items():
        if not str(num_str).isdigit():
            continue
        n = int(num_str)
        if not (1 <= n <= MAX_CORNER_NUM):
            continue
        try:
            a = float(coords[0])
            b = float(coords[1])
        except (TypeError, IndexError, ValueError):
            continue
        # Coords may come back as [y, x] in 0-1000 (preferred) or as [x, y] in
        # raw pixels of the sent image (legacy).  Detect which form we got.
        if max(a, b) <= 1010:
            cy = int(round(a / 1000.0 * h))
            cx = int(round(b / 1000.0 * w))
        else:
            cx = int(round(a / scale))
            cy = int(round(b / scale))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        r = find_ring_radius(gray, cx, cy, w, h)
        circles.append((cx, cy, r, num_str))
        all_circles.append((cx, cy, r))

    print(f"    Gemini detected corners: {sorted(int(c[3]) for c in circles)}")
    return circles, all_circles


# ── Font helper ───────────────────────────────────────────────────────────────

def best_font(size_px):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size_px)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_circuit_image(input_path, output_path, gemini_key=None, add_numbers=False):
    print(f"  Processing: {os.path.basename(input_path)}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"  ERROR: could not read {input_path}")
        return False

    # 1. Upscale low-resolution images so output lines are smooth, not blocky.
    #    Remember the original size so we can downscale the result back to it.
    orig_h, orig_w = img_bgr.shape[:2]
    img_bgr, _up_scale = upscale_if_needed(img_bgr)

    h, w = img_bgr.shape[:2]

    # 2. Normalise non-white backgrounds to white (handles d.jpeg-style gray canvas)
    img_bgr = normalize_background(img_bgr)

    # 3. Auto-select dark threshold; build raw dark mask
    raw_dark, threshold = make_dark_mask(img_bgr)
    print(f"    Dark threshold: {threshold}")

    # 4. Clean dark mask — removes text, gray-track borders, off-circuit titles.
    #    We intentionally *do not* try to resurrect tiny components here, because
    #    that can reintroduce random circles near the track.
    dark_clean = keep_circuit_components(raw_dark)

    # 5. Circle/number handling is optional.
    #    In default cleaning mode (add_numbers=False), skip all circle logic entirely.
    #    This preserves map symbols (e.g. start/finish checker marks), avoids ring
    #    artifacts, and never calls Gemini.
    hough_digit_circles = []   # (x, y, r_inner, r_outer, num_or_None)
    circles = []               # legacy text-rendering path (unused now)
    all_circles = []           # (x, y, r_inner, r_outer) for ring erasure
    if add_numbers:
        # Local-only pipeline: detect rings + OCR each one.  No Gemini.
        numbered_strict, ontrack_circles = detect_circled_numbers(img_bgr, dark_clean)
        loose_rings = find_all_rings(img_bgr)
        loose_rings = keep_rings_near_track(
            loose_rings, dark_clean, halo_px=36, min_overlap=8
        )

        # Dedup combined ring list at ~40 px on the upscaled image.
        dedup_d2 = max(40, int(_up_scale * 12)) ** 2
        raw_rings = []
        for xr, yr, rr in list(ontrack_circles) + list(loose_rings):
            if not any((xr - ex) ** 2 + (yr - ey) ** 2 < dedup_d2
                       for ex, ey, _ in raw_rings):
                raw_rings.append((xr, yr, rr))

        # Seed labels from the strict OCR pass (already deduped per-number).
        label_seed = {(x, y): num for x, y, r, num in numbered_strict}

        # Filter out tiny "rings" — corner markers are all roughly the same size,
        # so any ring much smaller than the median is a Hough false positive
        # (digit-internal loop, small map symbol, anti-aliasing artifact).
        orig_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if raw_rings:
            median_r = float(np.median([r for _, _, r in raw_rings]))
            min_keep = max(int(round(median_r * 0.6)), 12)
            raw_rings = [(x, y, r) for x, y, r in raw_rings if r >= min_keep]

        # Hough's max-radius cap (50 px) understates real ring size on upscaled
        # images.  Scale RING_PAD with the upscale factor so the erasure covers
        # the full ring stroke even when Hough underestimates the radius.
        RING_PAD = max(6, int(round(_up_scale * 8)))

        # First pass — collect rings with their OCR readings + confidence.
        with_conf = []  # (x, y, r, num_or_None, conf)
        for x, y, r in raw_rings:
            num, conf = None, 0.0
            for (lx, ly), lnum in label_seed.items():
                if (lx - x) ** 2 + (ly - y) ** 2 <= max(r + 6, 14) ** 2:
                    num, conf = lnum, 1.0   # strict pass is trusted
                    break
            if num is None:
                num_try, c = _ocr_circle(orig_gray, x, y, max(r, 8), w, h)
                # Require a clear majority across OCR variants.  Lower confidence
                # falls back to pixel extraction (preserves the original digit
                # shape, which beats a confidently wrong Tesseract guess).
                if num_try is not None and c >= 0.5:
                    num, conf = num_try, c
            with_conf.append((x, y, r, num, conf))

        # Each unique number can label at most one ring — the highest-confidence
        # one.  This kills the runaway "1" misreads OCR loves to produce.
        best_by_num: dict[str, int] = {}
        for i, (_, _, _, num, conf) in enumerate(with_conf):
            if num is None:
                continue
            j = best_by_num.get(num)
            if j is None or conf > with_conf[j][4]:
                best_by_num[num] = i
        kept = set(best_by_num.values())

        for i, (x, y, r, num, conf) in enumerate(with_conf):
            if num is not None and i not in kept:
                num = None  # demote losers — fallback to pixel extraction
            r_inner = max(2, r - RING_PAD)
            r_outer = r + RING_PAD
            all_circles.append((x, y, r_inner, r_outer))
            hough_digit_circles.append((x, y, r, r_outer, num))

        labeled = sum(1 for _, _, _, _, n in hough_digit_circles if n is not None)
        print(f"    Total rings for erasure: {len(all_circles)} "
              f"({labeled} OCR-labeled)")

    # 7. Red markers filtered to circuit proximity, then by shape: keep solid
    #    direction arrows, drop thin red corner-marker arcs/rings.
    red_mask     = make_red_mask(img_bgr)
    red_filtered = filter_red_near_circuit(red_mask, dark_clean)
    red_filtered = drop_arc_shaped_red(red_filtered)

    # Restore arrow stroke thickness lost during upscaling.
    # LANCZOS4 blends arrow edge pixels with the white background, pushing them
    # below the red threshold and making strokes appear thinner at high res.
    # A small dilation (1 px per scale step) recovers those lost edge pixels.
    if _up_scale > 1.2:
        dil_r  = max(1, int(round(_up_scale * 0.6)))
        k_red  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dil_r + 1, 2 * dil_r + 1)
        )
        red_filtered = cv2.dilate(red_filtered, k_red)
        # Re-apply the circuit-proximity constraint so dilation doesn't leak
        # red pixels far from the track.
        circuit_prox = cv2.dilate(
            dark_clean,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (RED_PROXIMITY * 2 + 1, RED_PROXIMITY * 2 + 1)
            )
        )
        red_filtered = cv2.bitwise_and(red_filtered, circuit_prox)

    # 8. Build output: white background, black circuit, red markers
    result_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    result_rgb[dark_clean > 0]   = (0, 0, 0)
    result_rgb[red_filtered > 0] = (210, 0, 0)

    if add_numbers:
        # 9. Remove ring pixels from the output, preserving circuit line crossings.
        #
        #    Each ring has a precisely localized stroke band [r_inner, r_outer]
        #    from find_ring_band.  We erase that band entirely, then re-instate any
        #    dark pixel inside it that bridges circuit ink strictly inside AND
        #    strictly outside the band — those are track entry/exit crossings.
        k_bridge_side = 15
        k_bridge      = np.ones((k_bridge_side, k_bridge_side), np.uint8)
        for x, y, r_in, r_out in all_circles:
            outer_mask = np.zeros((h, w), np.uint8)
            cv2.circle(outer_mask, (x, y), r_out, 255, -1)
            inner_mask = np.zeros((h, w), np.uint8)
            cv2.circle(inner_mask, (x, y), max(1, r_in), 255, -1)
            ring_band = (outer_mask > 0) & (inner_mask == 0)

            inside_dark  = (inner_mask  > 0) & (dark_clean > 0)
            outside_dark = (outer_mask == 0) & (dark_clean > 0)

            inside_nbr  = cv2.dilate(inside_dark.astype(np.uint8),  k_bridge) > 0
            outside_nbr = cv2.dilate(outside_dark.astype(np.uint8), k_bridge) > 0

            # Dark pixels in the ring band that bridge circuit inside↔outside
            circuit_bridge = ring_band & (dark_clean > 0) & inside_nbr & outside_nbr

            result_rgb[ring_band]      = (255, 255, 255)
            result_rgb[circuit_bridge] = (0, 0, 0)
            # Also clear the interior — we'll redraw the digit cleanly below.
            result_rgb[inner_mask > 0] = (255, 255, 255)

        # 9b. Remove black annotation arrows — small isolated dark blobs near a
        #     ring (e.g. the leader arrow that points from text to a corner).
        #     Red pixels are EXCLUDED from this pass so circuit direction arrows
        #     and the start-line chevron are never touched.
        ARROW_HALO = 80   # px beyond the ring to search for annotation blobs

        rgb_r = result_rgb[..., 0]
        rgb_g = result_rgb[..., 1]
        rgb_b = result_rgb[..., 2]
        is_red_mask = (rgb_r > 150) & (rgb_g < 80) & (rgb_b < 80)
        result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
        result_dark = ((result_gray < 128) & ~is_red_mask).astype(np.uint8) * 255
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            result_dark, connectivity=8
        )

        if n_cc > 1:
            max_area    = int(np.max(cc_stats[1:, cv2.CC_STAT_AREA]))
            circuit_min = max_area * 0.10   # protect anything ≥ 10% of the biggest blob
        else:
            circuit_min = 0

        for x, y, r_in, r_out in all_circles:
            r = r_out
            arrow_max_area = max(1500, r * r * 6)   # scales with circle size

            halo_mask = np.zeros((h, w), np.uint8)
            cv2.circle(halo_mask, (x, y), r + ARROW_HALO, 255, -1)
            cv2.circle(halo_mask, (x, y), r + 3,          0,   -1)  # exclude ring itself

            for lbl in range(1, n_cc):
                area = cc_stats[lbl, cv2.CC_STAT_AREA]
                if area >= circuit_min:          # protect all large circuit segments
                    continue
                if area >= arrow_max_area:
                    continue
                component_mask = (cc_labels == lbl).astype(np.uint8) * 255
                if np.any(halo_mask & component_mask):
                    result_rgb[component_mask > 0] = (255, 255, 255)

    # 10. Re-paint digit pixels in red from ring interiors.
    #     Two-pass to avoid losing edits between PIL (text) and numpy (pixels):
    #       Pass A: pixel extraction for rings without OCR labels (numpy edits)
    #       Pass B: bold red text for rings with OCR labels (PIL edits)
    result_pil = Image.fromarray(result_rgb)
    if add_numbers and hough_digit_circles:
        orig_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        digit_thr = min(max(threshold + 45, 125), 190)
        bold_radius = max(1, int(round(_up_scale * 0.9)))
        k_bold = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bold_radius + 1, 2 * bold_radius + 1)
        )

        # Pass A — extract & redraw digit pixels for unlabeled rings.
        for x, y, r_in, r_out, num in hough_digit_circles:
            if num is not None:
                continue
            dark_digit_mask = extract_digit_pixels_in_circle(
                orig_gray, x, y, max(3, r_in), digit_thr
            )
            if np.sum(dark_digit_mask > 0) >= 2:
                dark_digit_mask = cv2.dilate(dark_digit_mask, k_bold)
                result_rgb[dark_digit_mask > 0] = (210, 0, 0)

        # Pass B — bold red text for OCR-labeled rings, drawn via PIL.
        result_pil = Image.fromarray(result_rgb)
        draw       = ImageDraw.Draw(result_pil)
        for x, y, r_in, r_out, num in hough_digit_circles:
            if num is None:
                continue
            font_size = max(18, int(round(r_in * 1.9)))
            font      = best_font(font_size)
            stroke_w  = max(2, font_size // 18)
            bb        = draw.textbbox((0, 0), num, font=font,
                                       stroke_width=stroke_w)
            tw, th    = bb[2] - bb[0], bb[3] - bb[1]
            draw.text(
                (x - tw // 2 - bb[0], y - th // 2 - bb[1]),
                num,
                fill=(210, 0, 0),
                font=font,
                stroke_width=stroke_w,
                stroke_fill=(210, 0, 0),
            )
        result_rgb = np.array(result_pil)

    # 11. Remove small white "holes" inside the black circuit (corner-marker dots
    #     at turn entries/exits).  These are small enclosed white components
    #     surrounded predominantly by black track.  Skipped when --add-numbers is
    #     off so the default cleanup preserves all original map symbols.
    if add_numbers:
        result_gray2 = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
        rgb_r = result_rgb[..., 0]
        rgb_g = result_rgb[..., 1]
        rgb_b = result_rgb[..., 2]
        is_black = (result_gray2 < 80) & (rgb_r < 80)
        is_red   = (rgb_r > 150) & (rgb_g < 80) & (rgb_b < 80)
        is_white = (result_gray2 > 200).astype(np.uint8) * 255

        # Flood-fill from each corner that's white to mark the outer background.
        ff       = is_white.copy()
        ff_mask  = np.zeros((h + 2, w + 2), np.uint8)
        for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if ff[sy, sx] > 0:
                cv2.floodFill(ff, ff_mask, (sx, sy), 0)
        # `ff` now holds only enclosed (non-background) white regions.

        n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(ff, connectivity=8)
        # Marker-dot area scales with upscale; cap generously.
        max_dot_area = int(max(2500, (HOUGH_MAX_R * _up_scale) ** 2 * 1.2))
        dilate_k     = np.ones((5, 5), np.uint8)
        for i in range(1, n_h):
            area = int(st_h[i, cv2.CC_STAT_AREA])
            if area < 6 or area > max_dot_area:
                continue
            comp = (lab_h == i)
            comp_u8  = comp.astype(np.uint8) * 255
            dil      = cv2.dilate(comp_u8, dilate_k) > 0
            boundary = dil & ~comp
            b_black  = int(np.sum(boundary & is_black))
            b_red    = int(np.sum(boundary & is_red))
            # Surrounded mostly by black, not red — i.e. a marker hole, not a
            # closed digit loop (interior of 0, 6, 8, 9).
            if b_black >= 10 and b_red < b_black * 0.25:
                result_rgb[comp] = (0, 0, 0)

        result_pil = Image.fromarray(result_rgb)

    # Save at the full processing resolution (upscaled).  We intentionally do
    # NOT downscale back to the original dimensions — keeping the larger canvas
    # preserves all the sub-pixel detail recovered during upscaling.
    # PNG gives lossless output; fall back to maximum-quality JPEG for other exts.
    _, ext = os.path.splitext(output_path)
    if ext.lower() in (".png",):
        result_pil.save(output_path, optimize=True)
    else:
        result_pil.save(output_path, quality=98, subsampling=0)
    print(f"    Saved {w}×{h}: {os.path.basename(output_path)}")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def collect_inputs(paths):
    if paths:
        return paths
    found = []
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        found.extend(glob.glob(pattern))
    # Exclude any previously cleaned outputs
    return [f for f in found if "_cleaned" not in os.path.splitext(f)[0]]


def main():
    parser = argparse.ArgumentParser(
        description="Clean circuit map images: remove text, gray sections, replace corner rings."
    )
    parser.add_argument(
        "--gemini-key",
        metavar="KEY",
        default=None,
        help="Gemini API key. Get one free at https://aistudio.google.com/app/apikey. "
             "Can also be set via the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--add-numbers",
        action="store_true",
        default=False,
        help="Draw corner numbers on the cleaned output (disabled by default).",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image files to process. If omitted, all JPEG/PNG files in cwd are processed.",
    )
    args = parser.parse_args()

    # Kept for backwards compatibility, but numbering now uses local ring
    # detection and does not require Gemini.
    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")

    inputs = collect_inputs(args.images)
    if not inputs:
        print("No images found. Pass filenames as arguments or run in a folder with images.")
        sys.exit(1)

    if args.add_numbers:
        print("Corner detection: Local closed-ring detection (no Gemini)\n")
        print("Number drawing: ENABLED (--add-numbers)\n")
    else:
        print("Corner detection: SKIPPED (clean-only mode)\n")
        print("Number drawing: SKIPPED (default)\n")

    print(f"Found {len(inputs)} image(s) to process.\n")
    ok = 0
    for path in inputs:
        name, ext = os.path.splitext(path)
        out = f"{name}_cleaned{ext}"
        if clean_circuit_image(path, out, gemini_key=gemini_key,
                               add_numbers=args.add_numbers):
            ok += 1

    print(f"\nDone. {ok}/{len(inputs)} image(s) cleaned successfully.")


if __name__ == "__main__":
    main()
