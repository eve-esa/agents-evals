"""Download Earth-Bench task rasters.

Tool code and the task JSON are already in this package. This module fetches
the GeoTIFF directories from the Hugging Face dataset pinned in
``upstream.lock.toml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from agentic_evals.paths import EARTH_BENCH_DATA, REPO_ROOT

MODALITY_QUESTIONS = {
    "spectrum": range(1, 101),
    "products": range(101, 189),
    "rgb": range(189, 249),
    "all": range(1, 249),
}


def questions(args) -> list[int]:
    """Task numbers whose rasters to download. Empty when ``--skip-data``."""
    if args.skip_data:
        return []
    selected: list[int] = []
    if args.modality:
        selected.extend(MODALITY_QUESTIONS[args.modality])
    if args.questions:
        selected.extend(_parse_questions(args.questions))
    if not selected:
        selected.extend(MODALITY_QUESTIONS["all"])
    return sorted(set(selected))


def _parse_questions(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def install(args) -> None:
    selected = questions(args)
    if not selected:
        print("Earth-Bench rasters skipped (--skip-data). Tool code is already in the package.", flush=True)
        return
    from huggingface_hub import snapshot_download

    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    spec = lock["earth_bench_data"]
    target = EARTH_BENCH_DATA / "benchmark" / "data"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Earth-Bench rasters for {len(selected)} tasks.", flush=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=target,
        allow_patterns=[f"question{q}/*" for q in selected],
    )
    print(f"Rasters for {len(selected)} tasks in {target}", flush=True)
