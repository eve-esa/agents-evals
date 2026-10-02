"""Earth-Bench's five tool kits, served over MCP stdio.

The kits live in this package (ported from Earth-Agent ``agent/tools``; see
``PORTING.md``). They run as scripts under this interpreter, with the Earth-Bench
data directory as the working directory, so the ``benchmark/data/questionN``
paths in the prompts resolve. Every sample gets its own ``--temp_dir``.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from functools import cache
from pathlib import Path
from typing import Any

from inspect_ai.solver import TaskState

from agentic_evals.paths import EARTH_BENCH_DATA, RUNS_DIR

KITS = ("Analysis", "Index", "Inversion", "Perception", "Statistics")

# Returned as a normal observation by Perception's get_model_output when its
# precomputed lookup fails (bare ``except: pass``); see AGENTS.md.
PERCEPTION_FAILURE = "Failed to call model"

# Vendored lookup. Perception.py reads the same file via its own __file__.
PERCEPTION_LOOKUP = Path(__file__).resolve().parent / "model_results.csv"
_TOOLS_DIR = Path(__file__).resolve().parent


def ported_code_sha256() -> str:
    """Hash of the kit sources, so a run records which port produced it."""
    digest = hashlib.sha256()
    for path in sorted(_TOOLS_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kit_servers(output_dir: Path, *, root: Path = EARTH_BENCH_DATA) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    missing = [kit for kit in KITS if not (_TOOLS_DIR / f"{kit}.py").is_file()]
    if missing:
        raise FileNotFoundError(f"Earth-Bench kits missing from {_TOOLS_DIR}: {', '.join(missing)}")
    return {
        kit: {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(_TOOLS_DIR / f"{kit}.py"), "--temp_dir", str(output_dir)],
            "cwd": str(root),
            "env": {
                "FASTMCP_LOG_LEVEL": "WARNING",
                "PYTHONWARNINGS": "ignore",
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
    directory = tools_dir if tools_dir is not None else _TOOLS_DIR
    return _scalar_str_params(str(directory))


@cache
def _scalar_str_params(tools_dir: str) -> dict[str, frozenset[str]]:
    directory = Path(tools_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Earth-Agent tool sources not found at {directory}; "
            "the Earth-Bench kits are part of this package and should be here"
        )
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.py")):
        if path.name == "__init__.py":
            continue
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

    Spectrum and Products do not call Perception. RGB does, and without the
    vendored lookup every perception call returns ``Failed to call model``.
    """
    if modality != "rgb" or perception_lookup_available():
        return None
    return (
        "RGB is not applicable: Perception's get_model_output reads "
        f"{PERCEPTION_LOOKUP}, which is not on this machine, and would return "
        f"{PERCEPTION_FAILURE!r} for every call"
    )
