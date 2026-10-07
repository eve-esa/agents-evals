"""The Earth-Bench kits are the ported sources and serve the paper's 104 tools."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from agentic_evals.paths import EARTH_BENCH_DATA
from agentic_evals.tasks.earth_bench.tools import (
    KITS,
    PERCEPTION_LOOKUP,
    TOOL_RESPONSE_TIMEOUT,
    is_tool_failure,
    kit_servers,
    perception_lookup_available,
    rgb_unavailable_reason,
)
import agentic_evals.tasks.earth_bench.tools as earth_tools

pytestmark = pytest.mark.upstream


def test_perception_lookup_is_the_vendored_csv():
    assert PERCEPTION_LOOKUP.name == "model_results.csv"
    assert perception_lookup_available()
    text = PERCEPTION_LOOKUP.read_text(encoding="utf-8")
    assert "MSCN" in text
    assert "question189" in text


def test_rgb_is_unscored_only_when_the_lookup_is_missing(monkeypatch):
    monkeypatch.setattr(
        "agentic_evals.tasks.earth_bench.tools.perception_lookup_available",
        lambda: False,
    )
    assert rgb_unavailable_reason("rgb")
    assert rgb_unavailable_reason("spectrum") is None
    assert rgb_unavailable_reason("products") is None
    monkeypatch.setattr(
        "agentic_evals.tasks.earth_bench.tools.perception_lookup_available",
        lambda: True,
    )
    assert rgb_unavailable_reason("rgb") is None


def test_kit_servers_isolate_stdout_and_bound_the_read(tmp_path):
    servers = kit_servers(tmp_path, root=tmp_path)
    guard = Path(earth_tools.__file__).resolve().parent / "stdio_guard.py"
    analysis = servers["Analysis"]
    assert analysis["args"][0] == str(guard)
    assert analysis["args"][1].endswith("Analysis.py")
    assert analysis["session_kwargs"]["read_timeout_seconds"] == TOOL_RESPONSE_TIMEOUT
    assert TOOL_RESPONSE_TIMEOUT == timedelta(minutes=20)


@pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None, reason="uv sync --extra earth-bench"
)
def test_stdio_guard_keeps_prints_off_the_jsonrpc_line(tmp_path):
    """A print and a tool result must not share a stdout line.

    Reproduces the hang: mcp's stdio server writes JSON through a second
    wrapper on sys.stdout.buffer while print() uses the first. Without the
    guard those bytes become one line and the client never sees a response.
    """
    kit = tmp_path / "kit.py"
    kit.write_text(
        "\n".join(
            [
                "import os",
                "print('DIAGNOSTIC', flush=True)",
                "import anyio",
                "from mcp.server.stdio import stdio_server",
                "from mcp.shared.message import SessionMessage",
                "from mcp.types import JSONRPCMessage, JSONRPCRequest",
                "async def main():",
                "    async with stdio_server() as (_read, write):",
                "        message = JSONRPCMessage(JSONRPCRequest(jsonrpc='2.0', id=1, method='ping'))",
                "        await write.send(SessionMessage(message=message))",
                "        await anyio.sleep(0.1)",
                "        os._exit(0)",
                "anyio.run(main)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    guard = Path(earth_tools.__file__).resolve().parent / "stdio_guard.py"
    proc = subprocess.run(
        [sys.executable, str(guard), str(kit)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DIAGNOSTIC" not in proc.stdout
    assert proc.stdout.startswith("{")
    assert '"method":"ping"' in proc.stdout.replace(" ", "")
    assert proc.stdout.count("\n") == 1
    assert "DIAGNOSTIC" in proc.stderr


def test_is_tool_failure():
    assert is_tool_failure("Failed to call model")
    assert is_tool_failure({"text": "... Failed to call model ..."})
    assert not is_tool_failure("benchmark/out/question1/tvdi.tif")


@pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None, reason="uv sync --extra earth-bench"
)
def test_kits_serve_104_tools(tmp_path):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def run():
        client = MultiServerMCPClient(kit_servers(tmp_path))
        tools = await client.get_tools()
        by_name = {t.name: t for t in tools}
        listing = await by_name["get_filelist"].ainvoke(
            {"dir_path": "benchmark/data/question1"}
        )
        return tools, listing

    tools, listing = asyncio.run(run())
    assert len(KITS) == 5
    assert len(tools) == 104
    if (EARTH_BENCH_DATA / "benchmark" / "data" / "question1").exists():
        import json

        files = json.loads(listing[0]["text"])
        assert "Xinjiang_2019-01-01_LST.tif" in files
