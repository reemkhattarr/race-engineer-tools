# race-engineer-tools — task runner.
# `just` with no arguments lists all available recipes.

set dotenv-load := true

default:
    @just --list

# Install/sync the entire workspace (all members + dev tools) from uv.lock.
install:
    uv sync --all-packages

# Install git pre-commit hooks.
hooks:
    uv run pre-commit install

# Lint and format-check (no changes written) — CI parity.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint issues and format the codebase.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Static type checking.
types:
    uv run pyright

# Run the test suite.
test:
    uv run pytest

# Full local CI gate: lint + types + tests.
check: lint types test

# --- Tool 1: Circuit Map Cleaner ----------------------------------------------

# Launch the Gradio demo locally (upload a circuit map, clean it, download).
demo:
    uv run python -m circuit_cleaner.serving.demo

# Clean image(s) from the CLI, e.g. `just clean circuit-cleaner/assets/samples/a.jpeg`.
clean *IMAGES:
    uv run clean-circuit {{IMAGES}}
