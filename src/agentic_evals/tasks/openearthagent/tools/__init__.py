"""Per-sample MCP server config for OpenEarthAgent.

A live end-to-end run starts the cloned tool server when the local
controller is down (see ``tools/server.py``). This process proxies it.
``OPENEARTHAGENT_CONTROLLER_URL`` is the controller base, default
``http://127.0.0.1:20001``. Step mode does not start it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from inspect_ai.solver import TaskState

from agentic_evals.paths import OPENEARTHAGENT_DATA, OPENEARTHAGENT_GPKGS, OPENEARTHAGENT_IMAGES, RUNS_DIR

_TOOLS = Path(__file__).resolve().parent
_PROBE: dict[str, dict[str, Any]] = {}
TOOL_SERVICE_KEY = "agentic_evals:openearthagent_tool_service"
DEFAULT_CONTROLLER = "http://127.0.0.1:20001"


def controller_url(explicit: str = "") -> str:
    """Task argument, else ``OPENEARTHAGENT_CONTROLLER_URL``, else the upstream default."""
    return explicit or os.environ.get("OPENEARTHAGENT_CONTROLLER_URL", DEFAULT_CONTROLLER)


def gpkg_dir() -> Path:
    """Unpacked cache. ``OPENEARTHAGENT_GPKGS`` wins; otherwise ``data_paths.json``."""
    if "OPENEARTHAGENT_GPKGS" in os.environ:
        return OPENEARTHAGENT_GPKGS
    pointer = OPENEARTHAGENT_DATA / "data_paths.json"
    if pointer.is_file():
        recorded = json.loads(pointer.read_text(encoding="utf-8"))
        return Path(recorded["gpkgs"])
    return OPENEARTHAGENT_GPKGS


def ported_code_sha256() -> str:
    """Hash of this package's tool sources, recorded on the run."""
    digest = hashlib.sha256()
    for path in sorted(_TOOLS.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probe_controller(url: str) -> dict[str, Any]:
    """``POST /list_models``. Raises when the controller is down or lists nothing.

    Cached per URL for the process so a long run does not repeat the probe
    on every sample.
    """
    if url in _PROBE:
        return dict(_PROBE[url])
    try:
        response = httpx.post(url.rstrip("/") + "/list_models", timeout=10, trust_env=False)
        response.raise_for_status()
        models = response.json().get("models")
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"OpenEarthAgent tool controller not reachable at {url} ({exc}). "
            "Start the upstream tool server before an end-to-end run."
        ) from exc
    if not isinstance(models, list) or not models:
        raise RuntimeError(
            f"OpenEarthAgent controller at {url} listed no tools. "
            "Perception workers that are not registered fail the run instead of returning empty observations."
        )
    record = {"mode": "controller", "controller_url": url, "models": models}
    _PROBE[url] = record
    return dict(record)


def sample_servers_factory(controller: str):
    """MCP config for one sample. The process holds that sample's image and gpkg state."""

    def factory(state: TaskState) -> dict[str, dict[str, Any]]:
        images = [str(OPENEARTHAGENT_IMAGES / name) for name in (state.metadata.get("images") or [])]
        save = RUNS_DIR / "openearthagent" / str(state.sample_id).replace("/", "_")
        args = [
            "-m",
            "agentic_evals.tasks.openearthagent.tools.mcp_server",
            "--sample-num",
            str(state.metadata.get("idx", "")),
            "--images",
            ",".join(images),
            "--cache-dir",
            str(gpkg_dir()),
            "--save-dir",
            str(save),
            "--controller-url",
            controller,
        ]
        return {"openearthagent": {"transport": "stdio", "command": sys.executable, "args": args}}

    return factory
