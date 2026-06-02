"""Gradio demo for the circuit-map cleaner (tool #1 of race-engineer-tools).

A race engineer uploads a circuit-map image; the tool strips text, labels,
gray secondary-track regions and off-circuit clutter, leaving a clean white-
background outline (black track + red direction markers) that can be previewed
and downloaded. The broken --add-numbers corner-numbering path is never used:
we always call the cleaner with ``add_numbers=False``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from circuit_cleaner.clean_circuit import clean_circuit_image

# assets/samples lives at circuit-cleaner/assets/samples; this file is at
# circuit-cleaner/src/circuit_cleaner/serving/demo.py -> up 3 to the member root.
_SAMPLES_DIR = Path(__file__).resolve().parents[3] / "assets" / "samples"


def clean_uploaded_image(image_path: str | None) -> str:
    """Clean an uploaded circuit map and return the path to the cleaned image.

    ``image_path`` is the temp filepath Gradio hands us (``gr.Image`` with
    ``type="filepath"``). The underlying cleaner is disk-based, so we write the
    result to a fresh temp file — preserving the input extension so the encoder
    picks the right format — and return its path for preview + download.
    """
    if not image_path:
        raise gr.Error("Please upload a circuit-map image first.")

    suffix = os.path.splitext(image_path)[1] or ".png"
    fd, out_path = tempfile.mkstemp(suffix=f"_cleaned{suffix}")
    os.close(fd)

    ok = clean_circuit_image(image_path, out_path, add_numbers=False)
    if not ok:
        os.unlink(out_path)
        raise gr.Error("Could not process that image — is it a valid circuit map?")
    return out_path


def _sample_images() -> list[str]:
    if not _SAMPLES_DIR.is_dir():
        return []
    return sorted(str(p) for p in _SAMPLES_DIR.glob("*.jpeg"))


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI for the circuit cleaner."""
    with gr.Blocks(title="Circuit Map Cleaner") as demo:
        gr.Markdown(
            "# 🏁 Circuit Map Cleaner\n"
            "Upload a circuit-map image. The tool removes text, labels, gray "
            "secondary-track regions and off-circuit clutter, leaving a clean "
            "white-background outline (black track, red direction markers) you "
            "can download.\n\n"
            "_Tool #1 of the race-engineer-tools suite._"
        )
        with gr.Row():
            inp = gr.Image(
                type="filepath",
                label="Upload circuit map",
                sources=["upload"],
            )
            with gr.Column():
                out_img = gr.Image(type="filepath", label="Cleaned (preview)")
                out_file = gr.File(label="Download cleaned image")
        btn = gr.Button("Clean", variant="primary")

        def _run(path: str | None) -> tuple[str, str]:
            result = clean_uploaded_image(path)
            return result, result  # preview + download share the same file

        btn.click(_run, inputs=inp, outputs=[out_img, out_file])

        samples = _sample_images()
        if samples:
            gr.Examples(
                examples=[[s] for s in samples],
                inputs=inp,
                label="Example circuit maps",
            )
    return demo


if __name__ == "__main__":
    build_demo().launch()
