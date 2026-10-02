"""Earth-Agent's five tool kits, served unmodified over MCP stdio.

Each kit (``agent/tools/{Analysis,Index,Inversion,Perception,Statistics}.py``) is a
``fastmcp`` server that calls ``mcp.run()`` (stdio) and takes ``--temp_dir`` for its
outputs. They run in their own venv (``upstream.lock.toml``: numpy 1.24, Python
3.11, osgeo.gdal) with the Earth-Agent root as working directory, so the
``benchmark/data/questionN`` paths in the prompts resolve. Every sample gets its
own ``--temp_dir``, so parallel samples cannot overwrite each other's outputs.
"""

from __future__ import annotations

import ast
import os
from functools import cache
from pathlib import Path
from typing import Any

from inspect_ai.solver import TaskState

from agentic_evals.paths import EARTH_AGENT_ROOT, EARTH_AGENT_VENV, RUNS_DIR, venv_python

KITS = ("Analysis", "Index", "Inversion", "Perception", "Statistics")

# Returned as a normal observation by Perception's get_model_output when its
# precomputed lookup fails (bare ``except: pass``); see AGENTS.md.
PERCEPTION_FAILURE = "Failed to call model"

# Literal path in agent/tools/Perception.py (get_model_output). The file is not
# in the Earth-Agent checkout; without it every RGB perception call returns
# PERCEPTION_FAILURE and the sample is not a measurement of the model.
PERCEPTION_LOOKUP = Path("/root/autodl-tmp/Earth-Agent/benchmark/model_results.csv")


def kit_servers(output_dir: Path, *, root: Path = EARTH_AGENT_ROOT, venv: Path = EARTH_AGENT_VENV) -> dict[str, dict[str, Any]]:
    python = venv_python(venv)
    if not python.exists():
        raise FileNotFoundError(
            f"Earth-Agent tools venv not found at {venv}; run `uv run python scripts/setup_benchmark.py earth-bench`"
        )
    return {
        kit: {
            "transport": "stdio",
            "command": str(python),
            "args": [f"agent/tools/{kit}.py", "--temp_dir", str(output_dir)],
            "cwd": str(root),
            # MCP stdio inherits only a small default env, not LD_LIBRARY_PATH.
            # osgeo is linked to libgdal.so.34, which is missing unless the
            # process that starts the eval exported that path (a GDAL prefix
            # outside the system linker path).
            "env": {
                "FASTMCP_LOG_LEVEL": "WARNING",
                "PYTHONWARNINGS": "ignore",
                **({"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]} if os.environ.get("LD_LIBRARY_PATH") else {}),
            },
        }
        for kit in KITS
    }


def sample_output_dir(state: TaskState) -> Path:
    path = RUNS_DIR / "earth_bench" / f"question{state.sample_id}" / f"epoch{state.epoch}-{state.uuid}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_servers(state: TaskState) -> dict[str, dict[str, Any]]:
    return kit_servers(sample_output_dir(state))


def scalar_str_params(tools_dir: Path | None = None) -> dict[str, frozenset[str]]:
    """Parameter names annotated exactly ``str`` in the published tool sources.

    Parsed with ``ast`` so the Earth-Agent package is not imported. A parameter
    annotated ``list[str]`` or ``str | list[str]`` is omitted: that tool already
    accepts a list, and the gold step that passes one stays a single call.
    """
    directory = tools_dir if tools_dir is not None else EARTH_AGENT_ROOT / "agent" / "tools"
    return _scalar_str_params(str(directory))


@cache
def _scalar_str_params(tools_dir: str) -> dict[str, frozenset[str]]:
    directory = Path(tools_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Earth-Agent tool sources not found at {directory}; "
            "run `uv run python scripts/setup_benchmark.py earth-bench --skip-data`"
        )
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {
                arg.arg
                for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                if arg.annotation is not None and ast.unparse(arg.annotation) == "str"
            }
            if names:
                found[node.name] = names
    return {name: frozenset(params) for name, params in found.items()}


def is_tool_failure(output: Any) -> bool:
    return PERCEPTION_FAILURE in str(output)


def perception_lookup_available() -> bool:
    return PERCEPTION_LOOKUP.is_file()


def rgb_unavailable_reason(modality: str) -> str | None:
    """Why an RGB sample cannot be scored on this machine, or None if it can.

    Spectrum and Products do not call Perception. RGB does, and the lookup
    table is a hardcoded path on the authors' machine (Feng et al., ICLR 2026;
    ``agent/tools/Perception.py``). A missing file means every perception call
    returns ``Failed to call model``, so the sample is left unscored.
    """
    if modality != "rgb" or perception_lookup_available():
        return None
    return (
        "RGB is not applicable: Perception's get_model_output reads "
        f"{PERCEPTION_LOOKUP}, which is not on this machine, and would return "
        f"{PERCEPTION_FAILURE!r} for every call"
    )
