"""GeoBenchX's 24 tools, one MCP stdio process per sample.

The server is ``mcp_server.py`` in this package, run with this interpreter. It
holds the ``data_store`` that the tools pass dataframes through, so a process
must not be shared across samples. Dataset directories come from
``data_paths.json`` written by ``setup_benchmark.py geobenchx``, or from
``GEOBENCHX_STATDATA`` / ``GEOBENCHX_GEODATA``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from inspect_ai.solver import TaskState

from agentic_evals.paths import GEOBENCHX_DATA, GEOBENCHX_GEODATA, GEOBENCHX_STATDATA

_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_TOOLS_DIR = Path(__file__).resolve().parent


def data_directories() -> tuple[Path, Path]:
    """``(StatData, Geodata)``. Env overrides win; otherwise ``data_paths.json``."""
    stat, geo = GEOBENCHX_STATDATA, GEOBENCHX_GEODATA
    if stat.is_dir() and geo.is_dir():
        return stat, geo
    pointer = GEOBENCHX_DATA / "data_paths.json"
    if pointer.exists() and "GEOBENCHX_STATDATA" not in os.environ and "GEOBENCHX_GEODATA" not in os.environ:
        recorded = json.loads(pointer.read_text(encoding="utf-8"))
        return Path(recorded["stat"]), Path(recorded["geo"])
    return stat, geo


def judge_prompt() -> str:
    """The committed ``RESULT_CHECKING_PROMPT`` template."""
    path = _PROMPTS / "judge_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is part of this package and should be here.")
    return path.read_text(encoding="utf-8")


def ported_code_sha256() -> str:
    """Hash of the ported tool sources, so a run records which port produced it."""
    digest = hashlib.sha256()
    for path in sorted(_TOOLS_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sample_servers(state: TaskState) -> dict[str, dict[str, Any]]:
    """MCP config for this sample. Raises if the datasets are missing."""
    del state  # one server; the process itself isolates the sample's data_store
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
            "command": sys.executable,
            "args": ["-m", "agentic_evals.tasks.geobenchx.tools.mcp_server"],
            "cwd": str(GEOBENCHX_DATA),
            "env": {
                "GEOBENCHX_STATDATA": str(stat),
                "GEOBENCHX_GEODATA": str(geo),
                "STATDATAPATH": str(stat),
                "GEODATAPATH": str(geo),
                "MPLBACKEND": "Agg",
                "PYTHONWARNINGS": "ignore",
            },
        }
    }
