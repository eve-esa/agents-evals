"""GeoBenchX's 24 tools, one MCP stdio process per sample.

The server is ``scripts/geobenchx_mcp.py``, run with GeoBenchX's venv. It holds
the ``data_store`` that their tools pass dataframes through, so a process must
not be shared across samples. Dataset directories come from ``data_paths.json``
written by ``setup_benchmark.py geobenchx``, or from
``GEOBENCHX_STATDATA`` / ``GEOBENCHX_GEODATA``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from inspect_ai.solver import TaskState

from agentic_evals.paths import (
    GEOBENCHX_GEODATA,
    GEOBENCHX_JUDGE_PROMPT,
    GEOBENCHX_ROOT,
    GEOBENCHX_STATDATA,
    GEOBENCHX_VENV,
    REPO_ROOT,
    venv_python,
)


def data_directories() -> tuple[Path, Path]:
    """``(StatData, Geodata)``. Env overrides win; otherwise ``data_paths.json``."""
    stat, geo = GEOBENCHX_STATDATA, GEOBENCHX_GEODATA
    if stat.is_dir() and geo.is_dir():
        return stat, geo
    pointer = GEOBENCHX_ROOT / "data_paths.json"
    if pointer.exists() and "GEOBENCHX_STATDATA" not in os.environ and "GEOBENCHX_GEODATA" not in os.environ:
        recorded = json.loads(pointer.read_text(encoding="utf-8"))
        return Path(recorded["stat"]), Path(recorded["geo"])
    return stat, geo


def judge_prompt() -> str:
    """The exported ``RESULT_CHECKING_PROMPT`` template."""
    if not GEOBENCHX_JUDGE_PROMPT.exists():
        raise FileNotFoundError(
            f"{GEOBENCHX_JUDGE_PROMPT} not found. Build the GeoBenchX tools venv with "
            "`uv run python scripts/setup_benchmark.py geobenchx`."
        )
    return GEOBENCHX_JUDGE_PROMPT.read_text(encoding="utf-8")


def sample_servers(state: TaskState) -> dict[str, dict[str, Any]]:
    """MCP config for this sample. Raises if the venv or the datasets are missing."""
    del state  # one server; the process itself isolates the sample's data_store
    python = venv_python(GEOBENCHX_VENV)
    if not python.exists():
        raise FileNotFoundError(
            f"GeoBenchX tools venv not found at {GEOBENCHX_VENV}; "
            "run `uv run python scripts/setup_benchmark.py geobenchx`"
        )
    stat, geo = data_directories()
    missing = [path for path in (stat, geo) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "GeoBenchX dataset directories not found "
            f"({', '.join(str(path) for path in missing)}). "
            "Run `uv run python scripts/setup_benchmark.py geobenchx` "
            "or set GEOBENCHX_STATDATA and GEOBENCHX_GEODATA."
        )
    return {
        "geobenchx": {
            "transport": "stdio",
            "command": str(python),
            "args": [str(REPO_ROOT / "scripts" / "geobenchx_mcp.py")],
            "cwd": str(GEOBENCHX_ROOT),
            "env": {
                "GEOBENCHX_ROOT": str(GEOBENCHX_ROOT),
                "GEOBENCHX_STATDATA": str(stat),
                "GEOBENCHX_GEODATA": str(geo),
                "MPLBACKEND": "Agg",
                "PYTHONWARNINGS": "ignore",
                **({"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]} if os.environ.get("LD_LIBRARY_PATH") else {}),
            },
        }
    }
