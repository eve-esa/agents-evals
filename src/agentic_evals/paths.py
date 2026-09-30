"""Filesystem locations of fetched upstream benchmarks and run outputs.

Upstream repos are fetched by ``scripts/fetch_upstream.py`` into ``third_party/``.
Each location can be overridden with an environment variable so that hosted
infrastructure (e.g. the hands-on session) can point at pre-staged copies.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


THIRD_PARTY = _env_path("AGENTIC_EVALS_THIRD_PARTY", REPO_ROOT / "third_party")
EARTH_AGENT_ROOT = _env_path("EARTH_AGENT_ROOT", THIRD_PARTY / "Earth-Agent")
EARTH_AGENT_VENV = _env_path("EARTH_AGENT_VENV", THIRD_PARTY / "venvs" / "earth-agent")
EO_GYM_ROOT = _env_path("EO_GYM_ROOT", THIRD_PARTY / "EO-Gym")
EO_GYM_VENV = _env_path("EO_GYM_VENV", THIRD_PARTY / "venvs" / "eo-gym")
# Tool schemas and per-dataset tool subsets exported from EO-Gym's registry.
EO_GYM_TOOLS_JSON = _env_path("EO_GYM_TOOLS_JSON", THIRD_PARTY / "eo_gym_tools.json")
GEOBENCHX_ROOT = _env_path("GEOBENCHX_ROOT", THIRD_PARTY / "GeoBenchX")
GEOBENCHX_VENV = _env_path("GEOBENCHX_VENV", THIRD_PARTY / "venvs" / "geobenchx")
# Judge prompt exported from geobenchx.evaluation (tools description filled in).
GEOBENCHX_JUDGE_PROMPT = _env_path("GEOBENCHX_JUDGE_PROMPT", THIRD_PARTY / "geobenchx_judge_prompt.txt")
# Overridden when --with-data records where the Drive archive unpacked.
GEOBENCHX_STATDATA = _env_path("GEOBENCHX_STATDATA", GEOBENCHX_ROOT / "Data" / "StatData")
GEOBENCHX_GEODATA = _env_path("GEOBENCHX_GEODATA", GEOBENCHX_ROOT / "Data" / "Geodata")
RUNS_DIR = _env_path("AGENTIC_EVALS_RUNS_DIR", REPO_ROOT / "runs")


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"
