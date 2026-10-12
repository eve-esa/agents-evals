"""MCP stdio server in front of an OpenEarthAgent tool controller.

One process per sample. It keeps the image and GeoPackage handles, replays
the gpkg cache, and forwards everything else to the FastChat-style controller
the upstream launcher starts (``POST /get_worker_address``, then
``POST {worker}/worker_generate``). ``error_code != 0`` is an MCP error so
the trace records the failure, and the text is still the observation the
model sees.

Offline tools (``GetBboxFromGeotiff``, ``DisplayOnGeotiff``) are in-process
in the upstream tool server. They succeed here only when the controller
lists them as workers. Otherwise the call returns ``Tool {name} not found.``

The five gpkg tools are served from the cache on a hit (``error_code`` 0).
Every other result, including a cache miss, is sent to the controller.
That is ``BaseToolInferencer`` at the pinned revision.

Run with::

    python -m agentic_evals.tasks.openearthagent.tools.mcp_server \\
        --sample-num 0 --images /path/a.jpg --cache-dir data/openearthagent/gpkgs \\
        --save-dir runs/openearthagent/0 --controller-url http://127.0.0.1:20001
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import httpx
import mcp.server.stdio
from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from agentic_evals.tasks.openearthagent.tools.cache import cache_response
from agentic_evals.tasks.openearthagent.tools.catalog import schemas_for
from agentic_evals.tasks.openearthagent.tools.session import ToolSession, observe, prepare_call


def timeout_for(name: str) -> float:
    """``ToolManager.call_tool`` timeouts, in seconds. HTTP timeouts, not SIGALRM."""
    if name in {"AddPoisLayer", "ComputeDistance"}:
        return 180
    if name == "AddIndexLayer":
        return 300
    if name in {"ChangeDetection", "GetAreaBoundary"}:
        return 120
    return 60


async def call_worker(
    client: httpx.AsyncClient,
    controller: str,
    addresses: dict[str, str],
    name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """``ToolManager.call_tool`` for an online worker."""
    try:
        if name not in addresses:
            found = await client.post(controller.rstrip("/") + "/get_worker_address", json={"model": name})
            found.raise_for_status()
            address = (found.json() or {}).get("address") or ""
            if not address:
                return {"text": f"Tool {name} not found.", "error_code": 1}
            addresses[name] = address
        response = await client.post(addresses[name].rstrip("/") + "/worker_generate", json=params, timeout=timeout_for(name))
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"text": f"Tool {name} failed : {exc}", "error_code": 1}
    if not isinstance(body, dict):
        return {"text": f"Tool {name} failed : response was not an object", "error_code": 1}
    if "error_code" not in body:
        body["error_code"] = 1
    return body


def build_server(
    *,
    sample_num: str,
    images: list[str],
    cache_dir: Path,
    save_dir: Path,
    controller: str,
) -> Server:
    server: Server = Server("openearthagent")
    session = ToolSession(images=images, sample_num=sample_num)
    schemas = schemas_for()
    by_name = {schema["name"]: schema for schema in schemas}
    addresses: dict[str, str] = {}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=schema["name"], description=schema["description"], inputSchema=schema["parameters"])
            for schema in schemas
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name not in by_name:
            response: dict[str, Any] = {"text": f"There is no tool named {name}.", "error_code": 1}
        else:
            params, error = prepare_call(session, name, arguments or {})
            response = error or {"text": "", "error_code": 1}
            if error is None and params is not None:
                cached = cache_response(session, name, params, cache_dir=cache_dir, save_dir=save_dir)
                if cached is not None and cached.get("error_code") == 0:
                    response = cached
                elif not controller:
                    response = cached or {"text": f"Tool {name} not found.", "error_code": 1}
                else:
                    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                        response = await call_worker(client, controller, addresses, name, params)
        text = observe(session, name, response)
        failed = response.get("error_code") != 0
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=failed)

    return server


async def serve(args: argparse.Namespace) -> None:
    images = [part for part in (args.images or "").split(",") if part]
    server = build_server(
        sample_num=args.sample_num,
        images=images,
        cache_dir=Path(args.cache_dir),
        save_dir=Path(args.save_dir),
        controller=args.controller_url or "",
    )
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="openearthagent",
                server_version="1",
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-num", required=True)
    parser.add_argument("--images", default="")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--controller-url", default="")
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
