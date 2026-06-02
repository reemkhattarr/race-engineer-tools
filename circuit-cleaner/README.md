# Circuit Map Cleaner

Tool #1 of [race-engineer-tools](../README.md). Cleans circuit-map images:
removes text, labels, gray secondary-track regions and off-circuit clutter,
leaving a clean white-background outline — black track plus red direction
markers.

| Before | After |
| --- | --- |
| messy diagram with labels, gray tracks, annotations | white background, black outline, red direction markers |

## Use it

### Web demo (upload → clean → download)

```bash
just demo            # from the repo root
# or:
uv run python -m circuit_cleaner.serving.demo
```

Open the printed local URL, upload a circuit map (or pick an example), click
**Clean**, then download the result.

### CLI

```bash
just clean circuit-cleaner/assets/samples/a.jpeg     # writes a_cleaned.jpeg
# or directly, processing all images in the current directory:
uv run clean-circuit
```

## How it works

The core is `clean_circuit_image(input_path, output_path, add_numbers=False)`
in [`src/circuit_cleaner/clean_circuit.py`](src/circuit_cleaner/clean_circuit.py).
It upscales if needed, normalises the background to white, builds a dark mask
with an auto-selected threshold, keeps only the connected components that form
the circuit (dropping text and gray borders), filters red direction markers to
those near the track, and composes the clean output.

The `--add-numbers` corner-numbering path (OCR via Tesseract) is **unmaintained
and disabled** in the web tool and the default install. It lives behind the
optional `add-numbers` extra:

```bash
uv sync --extra add-numbers     # only if you need the (legacy) numbering CLI
```

## Deploy (Hugging Face Space)

The Space bundle lives in [`deploy/hf_space/`](deploy/hf_space/): `app.py`,
`requirements.txt` (lean — `gradio`, `opencv-python-headless`, `numpy`,
`pillow`), and a `README.md` with the Space frontmatter. Upload it together with
`src/circuit_cleaner/` and `assets/samples/`, mirroring the repo layout so
`app.py`'s `sys.path` insert resolves.
