"""Gradio demo for the circuit-map cleaner (part of race-engineer-tools).

A race engineer uploads a circuit-map image; the tool strips text, labels,
gray secondary-track regions and off-circuit clutter, leaving a clean white-
background outline (black track + red direction markers) that can be previewed
and downloaded.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from circuit_cleaner.clean_circuit import DEFAULT, Settings, clean_circuit_image

# assets/samples lives at circuit-cleaner/assets/samples; this file is at
# circuit-cleaner/src/circuit_cleaner/serving/demo.py -> up 3 to the member root.
_SAMPLES_DIR = Path(__file__).resolve().parents[3] / "assets" / "samples"


def clean_uploaded_image(image_path: str | None, keep_thin_lines: bool = True) -> str:
    """Clean an uploaded circuit map and return the path to the cleaned PNG.

    ``image_path`` is the temp filepath Gradio hands us (``gr.Image`` with
    ``type="filepath"``). The result is written to a fresh temp PNG (lossless)
    and its path returned for preview + download.
    """
    if not image_path:
        raise gr.Error("Please upload a circuit-map image first.")

    fd, out_path = tempfile.mkstemp(suffix="_cleaned.png")
    os.close(fd)

    cfg = Settings(keep_thin_lines=keep_thin_lines, output_min_side=DEFAULT.output_min_side)
    if not clean_circuit_image(image_path, out_path, cfg):
        os.unlink(out_path)
        raise gr.Error("Could not process that image — is it a valid circuit map?")
    return out_path


def _sample_images() -> list[str]:
    if not _SAMPLES_DIR.is_dir():
        return []
    exts = ("*.jpeg", "*.jpg", "*.png", "*.webp")
    return sorted(str(p) for e in exts for p in _SAMPLES_DIR.glob(e))


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI for the circuit cleaner."""
    with gr.Blocks(title="Circuit Map Cleaner") as demo:
        gr.Markdown(
            "# 🏁 Circuit Map Cleaner\n"
            "Upload a circuit-map image. The tool removes text, labels, gray "
            "secondary-track regions and off-circuit clutter, leaving a clean "
            "white-background outline (black track, red direction markers) you "
            "can download. White, gray, coloured, dark or transparent backgrounds "
            "and any image size all work."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(
                    type="filepath",
                    label="Upload circuit map",
                    sources=["upload"],
                    # Pass the original file through untouched: the default
                    # image_mode="RGB" flattens transparent PNGs (usually to
                    # black), which the cleaner then can't recover from.
                    image_mode=None,
                    format="png",
                )
                thin = gr.Checkbox(value=True, label="Keep pit lanes / thin attached lines")
                btn = gr.Button("Clean", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="filepath", label="Cleaned", interactive=False)
                download = gr.DownloadButton(
                    "Download cleaned image", variant="secondary", visible=False
                )

        def _run(path: str | None, keep_thin: bool) -> tuple[str, gr.DownloadButton]:
            result = clean_uploaded_image(path, keep_thin)
            # Reveal the download button now that there's a file to fetch.
            return result, gr.DownloadButton(value=result, visible=True)

        btn.click(_run, inputs=[inp, thin], outputs=[out_img, download])

        samples = _sample_images()
        if samples:
            gr.Examples(examples=[[s] for s in samples], inputs=inp, label="Example circuit maps")
    return demo


if __name__ == "__main__":
    build_demo().launch()
