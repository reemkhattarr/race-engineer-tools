"""Assemble the deployable Hugging Face Space bundle for the Circuit Map Cleaner.

The Space mirrors the repo layout so `deploy/hf_space/app.py`'s
`sys.path.insert(0, "circuit-cleaner/src")` resolves unchanged. We copy the
hand-maintained Space files (app.py, requirements.txt, README.md) to the bundle
root, then the package and sample assets under `circuit-cleaner/`.

Run with `just build-space`, then deploy with `just deploy`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# circuit-cleaner/ (member root) — this file is at circuit-cleaner/deploy/build_space.py
MEMBER = Path(__file__).resolve().parents[1]
HF_SPACE = MEMBER / "deploy" / "hf_space"
BUILD = MEMBER / "deploy" / "space_build"


def build() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "circuit-cleaner" / "src").mkdir(parents=True)
    (BUILD / "circuit-cleaner" / "assets").mkdir(parents=True)

    # Root-level Space files.
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(HF_SPACE / name, BUILD / name)

    # Package + sample assets, mirroring the repo layout.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(
        MEMBER / "src" / "circuit_cleaner",
        BUILD / "circuit-cleaner" / "src" / "circuit_cleaner",
        ignore=ignore,
    )
    shutil.copytree(
        MEMBER / "assets" / "samples",
        BUILD / "circuit-cleaner" / "assets" / "samples",
    )
    return BUILD


if __name__ == "__main__":
    out = build()
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"Built Space bundle at {out} ({len(files)} files):")
    for f in files:
        print(f"  {f}")
