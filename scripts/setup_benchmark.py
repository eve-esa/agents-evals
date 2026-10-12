"""Install one benchmark's downloaded data.

The install steps live in each task's ``setup.py``. This script only checks
the name and calls that module.

    uv sync --extra earth-bench
    uv run python scripts/setup_benchmark.py earth-bench
    uv run python scripts/setup_benchmark.py earth-bench --modality spectrum
    uv sync --extra geobenchx
    uv run python scripts/setup_benchmark.py geobenchx
    uv run python scripts/setup_benchmark.py eo-gym
    uv run python scripts/setup_benchmark.py openearthagent
    uv run python scripts/setup_benchmark.py openearthagent --tasks 0-10

Earth-Bench without ``--modality`` or ``--questions`` downloads every task
raster (16.6 GB) into ``data/earth_bench``. GeoBenchX downloads the Google
Drive archive into ``data/geobenchx``. EO-Gym downloads the image archives and
lookup tables into ``data/eo_gym`` and installs the copied tool server into
``data/venvs/eo-gym``. The server is not started. OpenEarthAgent downloads
the test images and the gpkg cache into ``data/openearthagent`` and clones
the tool server there. The server is not started.

``--skip-data`` downloads nothing. For EO-Gym it still installs the server
virtualenv. For OpenEarthAgent it still clones the pinned tool server.
``generate_contours_display`` is the one GeoBenchX tool that still
needs a system GDAL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_evals.tasks.earth_bench.setup import MODALITY_QUESTIONS
from agentic_evals.tasks.earth_bench.setup import install as install_earth_bench
from agentic_evals.tasks.eo_gym.setup import install as install_eo_gym
from agentic_evals.tasks.geobenchx.setup import install as install_geobenchx
from agentic_evals.tasks.openearthagent.setup import install as install_openearthagent

BENCHMARKS = ("earth-bench", "eo-gym", "geobenchx", "openearthagent")
# The Earth-Bench tools were ported from the Earth-Agent repo. Accept the old name.
ALIASES = {"earth-agent": "earth-bench"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=[*BENCHMARKS, *ALIASES], help="Benchmark to install. The others are left untouched.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not download rasters, imagery, lookup tables, the Drive archive, or OpenEarthAgent images.",
    )
    parser.add_argument(
        "--tasks",
        help="OpenEarthAgent only. Dataset idx values whose images to download, e.g. '0-10,42'.",
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark = canonical_benchmark(args.benchmark)
    if benchmark != "earth-bench" and (args.modality or args.questions):
        sys.exit("--modality and --questions apply only to earth-bench.")
    if getattr(args, "tasks", None) and benchmark != "openearthagent":
        sys.exit("--tasks applies only to openearthagent.")
    if args.skip_data and (args.modality or args.questions or getattr(args, "tasks", None)):
        sys.exit("--skip-data downloads nothing. Drop it to use --modality, --questions, or --tasks.")
    print(f"Installing {benchmark} only.", flush=True)
    if benchmark == "earth-bench":
        install_earth_bench(args)
    elif benchmark == "eo-gym":
        install_eo_gym(args)
    elif benchmark == "geobenchx":
        install_geobenchx(args)
    else:
        install_openearthagent(args)


if __name__ == "__main__":
    main()
