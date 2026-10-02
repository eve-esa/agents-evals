"""Filesystem locations of benchmark data and run outputs.

Earth-Bench and GeoBenchX tool code lives in this package. Their bulk files
are downloaded by ``scripts/setup_benchmark.py`` into ``data/`` (gitignored).
EO-Gym stays a pinned checkout under ``third_party/``. Each location can be
overridden with an environment variable so hosted infrastructure can point at
a pre-staged copy.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("AGENTIC_EVALS_DATA", REPO_ROOT / "data")
# Working directory of the Earth-Bench kits. Prompts refer to benchmark/data/questionN.
EARTH_BENCH_DATA = _env_path("EARTH_BENCH_DATA", DATA_DIR / "earth_bench")
GEOBENCHX_DATA = _env_path("GEOBENCHX_DATA", DATA_DIR / "geobenchx")
GEOBENCHX_STATDATA = _env_path("GEOBENCHX_STATDATA", GEOBENCHX_DATA / "StatData")
GEOBENCHX_GEODATA = _env_path("GEOBENCHX_GEODATA", GEOBENCHX_DATA / "Geodata")

THIRD_PARTY = _env_path("AGENTIC_EVALS_THIRD_PARTY", REPO_ROOT / "third_party")
EO_GYM_ROOT = _env_path("EO_GYM_ROOT", THIRD_PARTY / "EO-Gym")
EO_GYM_VENV = _env_path("EO_GYM_VENV", THIRD_PARTY / "venvs" / "eo-gym")
EO_GYM_TOOLS_JSON = _env_path("EO_GYM_TOOLS_JSON", THIRD_PARTY / "eo_gym_tools.json")
RUNS_DIR = _env_path("AGENTIC_EVALS_RUNS_DIR", REPO_ROOT / "runs")


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"
