"""The Earth-Bench kits are the ported sources and serve the paper's 104 tools."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from agentic_evals.paths import EARTH_BENCH_DATA
from agentic_evals.tasks.earth_bench.tools import (
    KITS,
    PERCEPTION_LOOKUP,
    is_tool_failure,
    kit_servers,
    perception_lookup_available,
    rgb_unavailable_reason,
)

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


def test_is_tool_failure():
    assert is_tool_failure("Failed to call model")
    assert is_tool_failure({"text": "... Failed to call model ..."})
    assert not is_tool_failure("benchmark/out/question1/tvdi.tif")


@pytest.mark.skipif(importlib.util.find_spec("fastmcp") is None, reason="uv sync --extra earth-bench")
def test_kits_serve_104_tools(tmp_path):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def run():
        client = MultiServerMCPClient(kit_servers(tmp_path))
        tools = await client.get_tools()
        by_name = {t.name: t for t in tools}
        listing = await by_name["get_filelist"].ainvoke({"dir_path": "benchmark/data/question1"})
        return tools, listing

    tools, listing = asyncio.run(run())
    assert len(KITS) == 5
    assert len(tools) == 104
    if (EARTH_BENCH_DATA / "benchmark" / "data" / "question1").exists():
        import json

        files = json.loads(listing[0]["text"])
        assert "Xinjiang_2019-01-01_LST.tif" in files
