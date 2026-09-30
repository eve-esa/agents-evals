"""The Earth-Agent kits start from their own venv and serve the paper's 104 tools."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentic_evals.paths import EARTH_AGENT_VENV, venv_python
from agentic_evals.tasks.earth_bench.tools import (
    KITS,
    PERCEPTION_LOOKUP,
    is_tool_failure,
    kit_servers,
    rgb_unavailable_reason,
)

pytestmark = pytest.mark.upstream


def test_perception_lookup_path_is_the_upstream_literal(earth_agent_root):
    source = (earth_agent_root / "agent" / "tools" / "Perception.py").read_text(encoding="utf-8")
    assert str(PERCEPTION_LOOKUP) in source


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


@pytest.mark.skipif(not venv_python(EARTH_AGENT_VENV).exists(), reason="tools venv not built")
def test_kits_serve_104_tools(earth_agent_root, tmp_path):
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
    if (earth_agent_root / "benchmark" / "data" / "question1").exists():
        files = json.loads(listing[0]["text"])
        assert "Xinjiang_2019-01-01_LST.tif" in files
