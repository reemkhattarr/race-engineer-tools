"""Hugging Face Space entrypoint for the Circuit Map Cleaner demo.

The deployed Space mirrors the repo layout (`circuit-cleaner/...`) so the
package import below resolves unchanged. Bundle `circuit-cleaner/src/` and
`circuit-cleaner/assets/samples/` alongside this file when uploading.
"""

import sys

sys.path.insert(0, "circuit-cleaner/src")

from circuit_cleaner.serving.demo import build_demo

build_demo().launch()
