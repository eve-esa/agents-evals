"""MCP stdio server in front of EO-Gym's FastAPI tool server (``POST /execute``).

EO-Gym exposes its tools over HTTP, not MCP. This proxy serves a chosen subset of
them over MCP with the schemas EO-Gym's own driver builds (exported to JSON by
``scripts/export_eo_gym_tools.py``) and forwards each call to ``/execute``,
returning the observation text exactly as the driver's ``ToolAPIExecutor.execute``
would (including its ``Error: ...`` strings, which the released scorer counts as
illegal calls).

View-changing tools on the VHR-zoom dataset return a new crop; EO-Gym's
environment then shows it to the model as a new image
(``OpenAIFunctionCallEnv._append_image_update_message``). With
``--image-updates`` the proxy attaches that crop as MCP image content, together
with upstream's "Updated image URL ..." text, once per image.

Run with::

    python -m agentic_evals.tasks.eo_gym.tools.mcp_proxy --tools-json PATH \\
        --execute-url http://127.0.0.1:8081/execute --tools a,b,c [--image-updates] --data-root DIR
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import mcp.server.stdio
from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

IMAGE_UPDATE_TOOLS = {
    "crop_optical_or_sar_image",
    "zoom_out_optical_image",
    "move_up_optical_image",
    "move_down_optical_image",
    "move_left_optical_image",
    "move_right_optical_image",
}


def execute_output(response: httpx.Response) -> str:
    """Observation text as ``eo_gym_evaluation.tool_api.ToolAPIExecutor.execute`` returns it."""
    body = response.text
    if response.status_code >= 400:
        return f"Error: EO Tool API returned HTTP {response.status_code}: {body}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(payload, dict) and "output" in payload:
        output = payload["output"]
        return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    return json.dumps(payload, ensure_ascii=False, default=str)


def new_image_ref(tool_name: str, observation: str) -> str | None:
    """``driver._extract_new_image_url``."""
    if tool_name not in IMAGE_UPDATE_TOOLS:
        return None
    try:
        payload = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("new_image_url", "cropped_path", "cropped_image_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def local_image(ref: str, data_root: Path) -> Path | None:
    raw = ref[len("file://") :] if ref.startswith("file://") else ref
    for marker in ("EO_GYM_DATA", "image_cache"):
        if marker in raw:
            candidate = data_root / marker / raw.split(marker, 1)[1].lstrip("/")
            return candidate if candidate.exists() else None
    path = Path(raw)
    return path if path.exists() else None


def build_server(schemas: list[dict[str, Any]], execute_url: str, image_updates: bool, data_root: Path) -> Server:
    server: Server = Server("eo-gym-proxy")
    by_name = {s["name"]: s for s in schemas}
    seen_images: set[str] = set()
    cache: dict[str, list[types.ContentBlock]] = {}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=s["name"], description=s.get("description", ""), inputSchema=s["parameters"])
            for s in by_name.values()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        if name not in by_name:
            return [types.TextContent(type="text", text=f"Error: Tool '{name}' is not configured for API execution.")]
        key = json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
        if key in cache:
            return cache[key]
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(execute_url, json={"tool_name": name, "arguments": arguments or {}})
            observation = execute_output(response)
        except httpx.HTTPError as exc:
            observation = f"Error: EO Tool API call failed for {name}: {exc}"
        content: list[types.ContentBlock] = [types.TextContent(type="text", text=observation)]
        ref = new_image_ref(name, observation) if image_updates else None
        path = local_image(ref, data_root) if ref else None
        if path is not None and str(path) not in seen_images:
            seen_images.add(str(path))
            content.append(
                types.TextContent(
                    type="text",
                    text=f"Updated image URL (local) from {name}: {path}\nUse this as the current view for subsequent steps.",
                )
            )
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            content.append(types.ImageContent(type="image", data=base64.b64encode(path.read_bytes()).decode(), mimeType=mime))
        if not observation.strip().lower().startswith("error"):
            cache[key] = content
        return content

    return server


async def serve(args: argparse.Namespace) -> None:
    catalog = json.loads(Path(args.tools_json).read_text())
    wanted = set(args.tools.split(",")) if args.tools else set(catalog["all_tools"])
    schemas = [s for s in catalog["schemas"] if s["name"] in wanted]
    server = build_server(schemas, args.execute_url, args.image_updates, Path(args.data_root))
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="eo-gym-proxy",
                server_version="1",
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", required=True)
    parser.add_argument("--execute-url", required=True)
    parser.add_argument("--tools", default="")
    parser.add_argument("--image-updates", action="store_true")
    parser.add_argument("--data-root", default=".")
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
