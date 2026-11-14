# Circuit Map Cleaner

Part of [race-engineer-tools](../README.md). Cleans circuit-map images:
removes text, labels, corner-number rings, gray secondary-track regions, logos,
scale bars and other off-circuit clutter, leaving a clean white-background
outline — black track plus red direction markers.

Works on light or dark maps, any track colour, and any resolution (a 450 px
thumbnail and a 4000 px poster go through the same rules). A typical map takes
well under a second.

## Use it

### Web demo (upload → clean → download)

```bash
just demo  # from the repo root
# or:
uv run python -m circuit_cleaner.serving.demo
```

### CLI

```bash
uv run clean-circuit circuit-cleaner/assets/samples/a.jpeg   # writes a_cleaned.jpeg
uv run clean-circuit *.png --no-thin-lines --size 3000
```

`--no-thin-lines` drops pit lanes and other thin lines attached to the track;
`--size` sets the minimum output long side (default 2000, `0` = no upscaling).

### Python

```python
import cv2
from circuit_cleaner import Settings, clean_circuit

out = clean_circuit(cv2.imread("map.png"), Settings(keep_thin_lines=False))
```

## How it works

Everything lives in [`src/circuit_cleaner/clean_circuit.py`](src/circuit_cleaner/clean_circuit.py)
(~340 lines). There are no fixed pixel constants: the tool first measures the
track's stroke width **W** and expresses every size as a multiple of it.

1. **Background.** The border colour tells whether the map is light or dark;
   dark maps are inverted internally.
2. **Ink.** Pixels clearly darker than the background, split with Otsu so that
   lighter gray layouts and watermarks fall out on their own. Red is handled
   separately.
3. **Track.** A morphological opening with a disc of ~0.6 W erases everything
   drawn thinner than the track (text, rings, leader arrows, scale bars).
   Small gaps such as white corner dots are bridged, and the big surviving
   pieces are the track. Ink physically connected to it is kept, so details
   like start/finish flags survive.
4. **Pit lanes.** Thin lines that hang off the track and are long and straight
   are kept (toggle with `keep_thin_lines`).
5. **Red markers.** Red blobs are kept when they are solid (arrows, not
   rings/arcs) and within ~4 W of the track.
6. **Render.** Opacity comes from the source pixels' own darkness, so edges
   stay anti-aliased and upscaling the result gives smooth curves.

All knobs are in the `Settings` dataclass.

## Deploy (Hugging Face Space)

`just build-space` assembles the bundle from [`deploy/hf_space/`](deploy/hf_space/)
plus `src/circuit_cleaner/`; `just deploy` uploads it.
