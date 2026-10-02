"""MCP stdio server for the 24 tools GeoBenchX's agent is given.

Run with this interpreter. ``geobenchx_tools.py`` keeps dataframes in a
LangGraph ``InjectedState`` dict that is shared across calls of one task. This
process is started once per sample, so one module-level dict is that sample's
store. ``state`` is omitted from the tool schema and passed in here.

The model in upstream's agent sees the function's return string only. Maps are
side effects (``image_store`` / ``html_store``) and are not sent back; this
server returns the same string and then drops those stores so they do not
accumulate. ``visualize`` stays true, which is the branch their agent sets, and
matplotlib/plotly are forced off-screen so ``plt.show`` / ``fig.show`` do not
open a window.

``plot_contour_lines`` is defined upstream and is not in ``agent.py``'s tool
list, so it is not served.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import sys
from contextlib import asynccontextmanager

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("STATDATAPATH", os.environ.get("GEOBENCHX_STATDATA", ""))
os.environ.setdefault("GEODATAPATH", os.environ.get("GEOBENCHX_GEODATA", ""))

from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import anyio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import plotly.io as pio  # noqa: E402
from io import TextIOWrapper  # noqa: E402

from mcp import types  # noqa: E402
from mcp.server.lowlevel import NotificationOptions, Server  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402
from mcp.shared.message import SessionMessage  # noqa: E402

from agentic_evals.tasks.geobenchx.tools.geobenchx_tools import (  # noqa: E402
    analyze_raster_overlap_tool,
    calculate_column_statistics_tool,
    calculate_columns_tool,
    calculate_line_lengths_tool,
    create_buffer_tool,
    filter_categorical_tool,
    filter_numerical_tool,
    filter_points_by_raster_values_tool,
    generate_contours_display_tool,
    get_centroids_tool,
    get_raster_description_tool,
    get_raster_path_tool,
    get_unique_values_tool,
    get_values_from_raster_with_geometries_tool,
    load_data_tool,
    load_geodata_tool,
    make_bivariate_map_tool,
    make_choropleth_map_tool,
    make_heatmap_tool,
    merge_dataframes_tool,
    reject_task_tool,
    scale_column_by_value_tool,
    select_features_by_spatial_relationship_tool,
    visualize_geographies_tool,
)

# Their agent calls plt.show() / fig.show() after the return text is built.
plt.show = lambda *args, **kwargs: None  # type: ignore[method-assign]
pio.renderers.default = "json"

# geobenchx/agent.py ``tools`` list, in that order.
TOOLS = [
    load_data_tool,
    load_geodata_tool,
    make_choropleth_map_tool,
    make_bivariate_map_tool,
    merge_dataframes_tool,
    get_unique_values_tool,
    filter_categorical_tool,
    filter_numerical_tool,
    select_features_by_spatial_relationship_tool,
    filter_points_by_raster_values_tool,
    create_buffer_tool,
    get_raster_path_tool,
    get_raster_description_tool,
    get_values_from_raster_with_geometries_tool,
    analyze_raster_overlap_tool,
    calculate_line_lengths_tool,
    calculate_columns_tool,
    scale_column_by_value_tool,
    make_heatmap_tool,
    visualize_geographies_tool,
    get_centroids_tool,
    generate_contours_display_tool,
    reject_task_tool,
    calculate_column_statistics_tool,
]

# One process per sample. Tools mutate data_store in place.
STATE: dict[str, Any] = {
    "data_store": {},
    "image_store": [],
    "html_store": [],
    "messages": [],
    "visualize": True,
}


def input_schema(tool: Any) -> dict[str, Any]:
    """JSON schema of the tool, without the injected ``state`` argument."""
    schema = tool.args_schema.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    properties = schema.get("properties", {})
    properties.pop("state", None)
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name != "state"]
        if not schema["required"]:
            schema.pop("required")
    return schema


def build_server() -> Server:
    server: Server = Server("geobenchx")
    by_name = {tool.name: tool for tool in TOOLS}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=tool.name, description=tool.description or "", inputSchema=input_schema(tool))
            for tool in TOOLS
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        tool = by_name.get(name)
        if tool is None:
            return [types.TextContent(type="text", text=f"Error calling tool '{name}': unknown tool")]
        try:
            # The tool is synchronous. Running it on the event loop blocks stdin
            # close, so the client times out and tears the pipe down mid-write.
            result = await asyncio.to_thread(tool.func, **(arguments or {}), state=STATE)
        except Exception as exc:  # noqa: BLE001 - the model has to see the failure, as a tool result
            text = f"Error calling tool '{name}': {type(exc).__name__}: {exc}"
        else:
            text = result if isinstance(result, str) else str(result)
        STATE["image_store"].clear()
        STATE["html_store"].clear()
        return [types.TextContent(type="text", text=text)]

    return server


def is_disconnect(exc: BaseException) -> bool:
    """The client closed the stdio pipe. That is a finished sample, not a crash.

    ``mcp.server.stdio`` only swallows ``ClosedResourceError``. A flush after the
    client has gone raises ``BrokenPipeError`` from a worker thread, and anyio
    turns that into an exception group that kills this process.
    """
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(is_disconnect(item) for item in exc.exceptions)
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, anyio.BrokenResourceError, anyio.ClosedResourceError)):
        return True
    return isinstance(exc, OSError) and exc.errno == errno.EPIPE


@asynccontextmanager
async def _stdio_server():
    """``mcp.server.stdio.stdio_server``, plus a clean exit when the pipe closes."""
    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def stdin_reader() -> None:
        try:
            async with read_stream_writer:
                async for line in stdin:
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue
                    await read_stream_writer.send(SessionMessage(message))
        except BaseException as exc:
            if not is_disconnect(exc):
                raise
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await stdout.write(payload + "\n")
                    await stdout.flush()
        except BaseException as exc:
            if not is_disconnect(exc):
                raise
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        yield read_stream, write_stream


async def serve() -> None:
    server = build_server()
    async with _stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="geobenchx",
                server_version="1",
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )


def _exit_on_signal(signum: int, frame: object) -> None:
    del signum, frame
    # A blocked tool thread would ignore SystemExit and miss the client's
    # two-second shutdown wait. Leave immediately.
    os._exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)
    try:
        asyncio.run(serve())
    except BaseException as exc:
        if is_disconnect(exc):
            return
        raise


if __name__ == "__main__":
    main()
