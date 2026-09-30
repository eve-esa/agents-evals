"""Per-sample MCP server config for EO-Gym: our proxy over its ``/execute`` server.

EO-Gym's tool server must already be running (it needs the ~47 GB of data and,
for perception tools, a GPU): ``cd third_party/EO-Gym && scripts/start_tool_server.sh``.
The proxy runs in our environment (``sys.executable``) and exposes the tools the
driver would expose for the sample's dataset: the dataset-mapped "skill" subset,
or all 35.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx
from inspect_ai.solver import TaskState

from agentic_evals.paths import EO_GYM_ROOT, EO_GYM_TOOLS_JSON

VHR_ZOOM_DATASETS = {"vhr_zoom", "vhr-zoom", "vhrzoom", "fair1_vhr"}


def health_url(execute_url: str) -> str:
    return execute_url.rsplit("/", 1)[0] + "/healthz"


def check_tool_server(execute_url: str) -> None:
    try:
        response = httpx.get(health_url(execute_url), timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"EO-Gym tool server not reachable at {health_url(execute_url)} ({exc}). "
            "Start it with `cd third_party/EO-Gym && scripts/start_tool_server.sh` "
            "(needs the EO-Gym data; see AGENTS.md)."
        ) from exc


def proxy_servers(tools: list[str], data_source: str, execute_url: str) -> dict[str, dict[str, Any]]:
    args = [
        "-m", "agentic_evals.tasks.eo_gym.mcp_proxy",
        "--tools-json", str(EO_GYM_TOOLS_JSON),
        "--execute-url", execute_url,
        "--tools", ",".join(tools),
        "--data-root", str(EO_GYM_ROOT),
    ]
    if data_source.strip().lower() in VHR_ZOOM_DATASETS:
        args.append("--image-updates")
    return {"eo_gym": {"transport": "stdio", "command": sys.executable, "args": args}}


def sample_servers_factory(execute_url: str):
    def factory(state: TaskState) -> dict[str, dict[str, Any]]:
        return proxy_servers(state.metadata["tools"], state.metadata["data_source"], execute_url)

    return factory
