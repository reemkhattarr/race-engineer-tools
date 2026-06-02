"""Tests for the Gradio wrapper around the cleaner."""

import os
from pathlib import Path

import gradio as gr
import pytest

from circuit_cleaner.serving.demo import build_demo, clean_uploaded_image


def test_clean_uploaded_image_returns_existing_file(sample_image: Path) -> None:
    """The wrapper returns a path to a non-empty cleaned image."""
    result = clean_uploaded_image(str(sample_image))
    try:
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
    finally:
        if os.path.exists(result):
            os.unlink(result)


def test_clean_uploaded_image_rejects_empty_input() -> None:
    """No upload surfaces a user-facing gr.Error rather than crashing."""
    with pytest.raises(gr.Error):
        clean_uploaded_image(None)


def test_build_demo_constructs() -> None:
    """The Blocks app assembles without error (Space dry-run)."""
    demo = build_demo()
    assert isinstance(demo, gr.Blocks)
