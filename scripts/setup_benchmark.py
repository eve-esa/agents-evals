"""Install one benchmark's data. Tool code for Earth-Bench and GeoBenchX is in this package.

Pass the benchmark name. Nothing for the other benchmarks is downloaded.

    uv sync --extra earth-bench
    uv run python scripts/setup_benchmark.py earth-bench
    uv run python scripts/setup_benchmark.py earth-bench --modality spectrum
    uv sync --extra geobenchx
    uv run python scripts/setup_benchmark.py geobenchx
    uv run python scripts/setup_benchmark.py eo-gym

Earth-Bench without ``--modality`` or ``--questions`` downloads every task
raster (16.6 GB) into ``data/earth_bench``. GeoBenchX downloads the Google
Drive archive into ``data/geobenchx``. EO-Gym still fetches its checkout
(~47 GB with data) under ``third_party/``; its tool server is not started.

``--skip-data`` does nothing for Earth-Bench and GeoBenchX, whose code is
already installed, and skips EO-Gym's image archives. ``generate_contours_display``
is the one GeoBenchX tool that still needs a system GDAL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fetch_upstream  # noqa: E402

from agentic_evals.paths import (  # noqa: E402
    EARTH_BENCH_DATA,
    EO_GYM_ROOT,
    GEOBENCHX_DATA,
)

LOCK = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
MODALITY_QUESTIONS = {
    "spectrum": range(1, 101),
    "products": range(101, 189),
    "rgb": range(189, 249),
    "all": range(1, 249),
}

BENCHMARKS = ("earth-bench", "eo-gym", "geobenchx")
# The Earth-Bench tools live in the Earth-Agent repo. Accept the old name.
ALIASES = {"earth-agent": "earth-bench"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=[*BENCHMARKS, *ALIASES], help="Benchmark to install. The others are left untouched.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not download rasters, imagery, or the Drive archive.",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(MODALITY_QUESTIONS),
        help="Earth-Bench only. Download rasters for this modality instead of all 248 tasks.",
    )
    parser.add_argument("--questions", help="Earth-Bench only. Task numbers to download, e.g. '1-5,101'.")
    return parser


def canonical_benchmark(name: str) -> str:
    return ALIASES.get(name, name)


def earth_bench_questions(args: argparse.Namespace) -> list[int]:
    """Task numbers whose rasters to download. Empty when ``--skip-data``."""
    if args.skip_data:
        return []
    questions: list[int] = []
    if args.modality:
        questions.extend(MODALITY_QUESTIONS[args.modality])
    if args.questions:
        questions.extend(_parse_questions(args.questions))
    if not questions:
        questions.extend(MODALITY_QUESTIONS["all"])
    return sorted(set(questions))


def _parse_questions(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _find_dir(root: Path, name: str) -> Path | None:
    target = name.lower()
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def unpack_eo_gym_archives(root: Path = EO_GYM_ROOT) -> None:
    """Unpack EO-Gym's data zips when the matching directory is absent.

    Entries already under ``EO_GYM_DATA/`` or ``image_cache/`` extract into
    ``root``. A zip of bare files extracts into that directory.
    """
    for dirname in ("EO_GYM_DATA", "image_cache"):
        archive = root / f"{dirname}.zip"
        dest = root / dirname
        if dest.is_dir() or not archive.is_file():
            continue
        print(f"Unpacking {archive.name}", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            prefix = dirname + "/"
            if any(name.startswith(prefix) or name.startswith("./" + prefix) for name in zipped.namelist()):
                zipped.extractall(root)
            else:
                dest.mkdir(parents=True)
                zipped.extractall(dest)


def setup_earth_bench(args: argparse.Namespace) -> None:
    questions = earth_bench_questions(args)
    if not questions:
        print("Earth-Bench rasters skipped (--skip-data). Tool code is already in the package.", flush=True)
        return
    from huggingface_hub import snapshot_download

    spec = LOCK["earth_bench_data"]
    target = EARTH_BENCH_DATA / "benchmark" / "data"
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Earth-Bench rasters for {len(questions)} tasks.", flush=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=target,
        allow_patterns=[f"question{q}/*" for q in questions],
    )
    print(f"Rasters for {len(questions)} tasks in {target}", flush=True)


def setup_eo_gym(args: argparse.Namespace) -> None:
    fetch_upstream.fetch_eo_gym(with_data=not args.skip_data, build_venv=True)
    if args.skip_data:
        print("EO-Gym images skipped (--skip-data).", flush=True)
        return
    unpack_eo_gym_archives()
    print(
        "EO-Gym tool server is not started. From third_party/EO-Gym run scripts/start_tool_server.sh.",
        flush=True,
    )


def setup_geobenchx(args: argparse.Namespace) -> None:
    if args.skip_data:
        print("GeoBenchX Drive archive skipped (--skip-data). Tool code is already in the package.", flush=True)
        return
    spec = LOCK["geobenchx"]
    GEOBENCHX_DATA.mkdir(parents=True, exist_ok=True)
    dest = GEOBENCHX_DATA / "geobenchx_data.zip"
    if not dest.exists():
        subprocess.run(
            ["uvx", "gdown", f"https://drive.google.com/uc?id={spec['drive_file_id']}", "-O", str(dest)],
            check=True,
        )
    bundle = GEOBENCHX_DATA / "data_bundle"
    if not bundle.exists():
        with zipfile.ZipFile(dest) as archive:
            bundle.mkdir(parents=True)
            archive.extractall(bundle)
    stat = _find_dir(bundle, "StatData")
    geo = _find_dir(bundle, "Geodata")
    if stat is None or geo is None:
        sys.exit(f"Unpacked {dest} into {bundle}, but StatData and Geodata were not both found.")
    pointer = GEOBENCHX_DATA / "data_paths.json"
    pointer.write_text(json.dumps({"stat": str(stat.resolve()), "geo": str(geo.resolve())}, indent=2) + "\n")
    print(f"StatData: {stat}", flush=True)
    print(f"Geodata: {geo}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark = canonical_benchmark(args.benchmark)
    if benchmark != "earth-bench" and (args.modality or args.questions):
        sys.exit("--modality and --questions apply only to earth-bench.")
    if args.skip_data and (args.modality or args.questions):
        sys.exit("--skip-data downloads no rasters. Drop it to use --modality or --questions.")
    print(f"Installing {benchmark} only.", flush=True)
    if benchmark == "earth-bench":
        setup_earth_bench(args)
    elif benchmark == "eo-gym":
        setup_eo_gym(args)
    else:
        setup_geobenchx(args)


if __name__ == "__main__":
    main()
