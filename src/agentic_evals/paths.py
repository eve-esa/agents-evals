"""Filesystem locations of downloaded benchmark data and run outputs.

Committed task JSON lives inside each task package. Bulk files (rasters, the
GeoBenchX archive, EO-Gym imagery, OpenEarthAgent images and the cloned tool
server) are downloaded by ``scripts/setup_benchmark.py`` into ``data/``
(gitignored). Each location can be overridden with an environment variable so
hosted infrastructure can point at a pre-staged copy.
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

# Unpacked EO-Gym imagery (EO_GYM_DATA/ and image_cache/). The test split and
# tool schemas are committed under tasks/eo_gym.
EO_GYM_ROOT = _env_path("EO_GYM_ROOT", DATA_DIR / "eo_gym")
EO_GYM_VENV = _env_path("EO_GYM_VENV", DATA_DIR / "venvs" / "eo-gym")
EO_GYM_SERVER = _env_path(
    "EO_GYM_SERVER",
    REPO_ROOT / "src" / "agentic_evals" / "tasks" / "eo_gym" / "server",
)

# OpenEarthAgent test images (test/*.jpg) and the gpkg cache. The tool server
# is cloned into OPENEARTHAGENT_UPSTREAM by setup; it is not imported by this
# package.
OPENEARTHAGENT_DATA = _env_path("OPENEARTHAGENT_DATA", DATA_DIR / "openearthagent")
OPENEARTHAGENT_UPSTREAM = _env_path("OPENEARTHAGENT_UPSTREAM", OPENEARTHAGENT_DATA / "upstream")
OPENEARTHAGENT_IMAGES = _env_path("OPENEARTHAGENT_IMAGES", OPENEARTHAGENT_DATA / "test")
OPENEARTHAGENT_GPKGS = _env_path("OPENEARTHAGENT_GPKGS", OPENEARTHAGENT_DATA / "gpkgs")
RUNS_DIR = _env_path("AGENTIC_EVALS_RUNS_DIR", REPO_ROOT / "runs")


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"
