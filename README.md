# race-engineer-tools

A suite of small, focused utilities that assist a race engineer's day-to-day
work. These are practical tools — not ML-portfolio pieces — built to be run
locally or shipped as lightweight web demos.

> Sibling project to [`f1-ml`](../f1-ml), which holds the ML portfolio. This
> repo is deliberately separate: it's tooling for the engineer, not modelling.

## Circuit Map Cleaner — `circuit-cleaner/`

Turns a messy circuit-map image (text, labels, gray secondary tracks,
off-circuit clutter) into a clean white-background outline: black track plus red
direction markers, nothing else. Available as a CLI and as an interactive Gradio
app (upload → clean → download) deployable to a Hugging Face Space.

- Tool docs: [`circuit-cleaner/README.md`](circuit-cleaner/README.md)
- Run the web demo: `just demo`
- Clean from the CLI: `just clean circuit-cleaner/assets/samples/a.jpeg`

## Roadmap

Future tools will be added as workspace members alongside `circuit-cleaner`.
Candidates:

- **Pit-window calculator** — undercut/overcut timing from live gaps and deg.
- **Stint planner** — compound/stint-length scenarios for a given race length.
- **Telemetry note-taker** — quick annotated exports for debriefs.

(Placeholders — not yet built.)

## Development

This is a single [uv](https://docs.astral.sh/uv/) **workspace**: each tool is a
member with its own `pyproject.toml` sharing one root `uv.lock`. Python 3.11.

```bash
just install   # uv sync --all-packages
just hooks     # install pre-commit hooks
just check     # lint + types + tests (CI gate)
just demo      # launch the Circuit Map Cleaner locally
```

Tooling: `uv` (deps), `just` (commands), `ruff` (lint + format), `pyright`
(types), `pytest` (tests), `pre-commit` (hooks).
