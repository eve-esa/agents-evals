"""Install one benchmark: pinned checkout, tool environment, and its data.

Pass the benchmark name. Nothing for the other benchmarks is downloaded.

    uv run python scripts/setup_benchmark.py earth-bench
    uv run python scripts/setup_benchmark.py earth-bench --modality spectrum
    uv run python scripts/setup_benchmark.py eo-gym
    uv run python scripts/setup_benchmark.py geobenchx

Earth-Bench and GeoBenchX need a system GDAL (``gdal-config`` on ``PATH``).
Earth-Bench without ``--modality`` or ``--questions`` downloads every task
raster (16.6 GB). EO-Gym downloads and unpacks about 47 GB. GeoBenchX
downloads the Google Drive archive linked from upstream ``data/README.md``.

``--skip-data`` installs the checkout and the tool environment only. EO-Gym's
``/execute`` server is not started by this script; run
``third_party/EO-Gym/scripts/start_tool_server.sh`` after ``eo-gym``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_upstream as upstream  # noqa: E402

from agentic_evals.paths import EO_GYM_ROOT  # noqa: E402

BENCHMARKS = ("earth-bench", "eo-gym", "geobenchx")
# The Earth-Bench tools live in the Earth-Agent repo. Accept the old name.
ALIASES = {"earth-agent": "earth-bench"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=[*BENCHMARKS, *ALIASES], help="Benchmark to install. The others are left untouched.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Checkout and tool environment only. Do not download rasters, imagery, or the Drive archive.",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(upstream.MODALITY_QUESTIONS),
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
        questions.extend(upstream.MODALITY_QUESTIONS[args.modality])
    if args.questions:
        questions.extend(upstream.parse_questions(args.questions))
    if not questions:
        questions.extend(upstream.MODALITY_QUESTIONS["all"])
    return sorted(set(questions))


def require_gdal() -> None:
    if shutil.which("gdal-config") is None:
        sys.exit(
            "gdal-config not found. Earth-Bench and GeoBenchX import osgeo, which is built "
            "against a system GDAL (macOS: `brew install gdal`; Debian/Ubuntu: "
            "`apt-get install gdal-bin libgdal-dev`)."
        )


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
    require_gdal()
    upstream.fetch_earth_agent_code()
    upstream.build_earth_agent_venv()
    questions = earth_bench_questions(args)
    if not questions:
        print("Earth-Bench rasters skipped (--skip-data).", flush=True)
        return
    print(f"Downloading Earth-Bench rasters for {len(questions)} tasks.", flush=True)
    upstream.fetch_earth_bench_data(questions)


def setup_eo_gym(args: argparse.Namespace) -> None:
    upstream.fetch_eo_gym(with_data=not args.skip_data, build_venv=True)
    if args.skip_data:
        print("EO-Gym images skipped (--skip-data).", flush=True)
        return
    unpack_eo_gym_archives()
    print(
        "EO-Gym tool server is not started. From third_party/EO-Gym run scripts/start_tool_server.sh.",
        flush=True,
    )


def setup_geobenchx(args: argparse.Namespace) -> None:
    require_gdal()
    upstream.fetch_geobenchx(build_venv=True, with_data=not args.skip_data)
    if args.skip_data:
        print("GeoBenchX Drive archive skipped (--skip-data).", flush=True)


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
