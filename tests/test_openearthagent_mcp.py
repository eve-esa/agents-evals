"""MCP proxy: a cache hit stays local, and a miss calls the worker."""

from __future__ import annotations

import asyncio

import geopandas as gpd
import pytest
from mcp import types
from shapely.geometry import Point

from agentic_evals.tasks.openearthagent.tools.mcp_server import build_server, call_worker


def _tools(server):
    handler = server.request_handlers[types.ListToolsRequest]

    async def run():
        result = await handler(types.ListToolsRequest())
        return result.root.tools

    return asyncio.run(run())


def _call(server, name: str, arguments: dict):
    handler = server.request_handlers[types.CallToolRequest]

    async def run():
        request = types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments))
        result = await handler(request)
        return result.root

    return asyncio.run(run())


def test_list_tools_is_the_live_set(tmp_path):
    live = build_server(
        sample_num="0",
        images=[],
        cache_dir=tmp_path,
        save_dir=tmp_path / "save",
        controller="http://127.0.0.1:20001",
    )
    names = {tool.name for tool in _tools(live)}
    assert "Calculator" in names
    assert "GetAreaBoundary" in names
    assert "Terminate" not in names


def test_cache_hit_skips_the_worker_and_a_miss_calls_it(monkeypatch: pytest.MonkeyPatch, tmp_path):
    seen: list[str] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "live boundary", "error_code": 0, "address": "http://worker.test"}

    class _Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, timeout=None):
            del json, timeout
            seen.append(url)
            if url.endswith("/get_worker_address"):
                return _Response()
            if url.endswith("/worker_generate"):
                return _Response()
            raise AssertionError(url)

    monkeypatch.setattr("agentic_evals.tasks.openearthagent.tools.mcp_server.httpx.AsyncClient", _Client)
    cache = tmp_path / "cache"
    cache.mkdir()
    area = "Banff National Park"
    frame = gpd.GeoDataFrame(
        {"name": [f"{area}_buffer3000m"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    frame.to_file(cache / "0.gpkg", layer="area_boundary", driver="GPKG")
    server = build_server(
        sample_num="0",
        images=[],
        cache_dir=cache,
        save_dir=tmp_path / "save",
        controller="http://127.0.0.1:20001",
    )
    hit = _call(server, "GetAreaBoundary", {"area": area, "buffer_m": 3000})
    assert hit.isError is False
    assert hit.content[0].text.startswith("Saved boundary")
    assert seen == []
    miss = _call(server, "GetAreaBoundary", {"area": "Nowhere", "buffer_m": 10})
    assert miss.isError is False
    assert miss.content[0].text == "live boundary"
    assert any(url.endswith("/worker_generate") for url in seen)


def test_controller_call_uses_worker_generate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    seen: list[tuple[str, dict]] = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, timeout=None):
            del timeout
            seen.append((url, json))
            if url.endswith("/get_worker_address"):
                return _Response({"address": "http://worker.test"})
            if url.endswith("/worker_generate"):
                return _Response({"text": "4", "error_code": 0})
            raise AssertionError(url)

    monkeypatch.setattr("agentic_evals.tasks.openearthagent.tools.mcp_server.httpx.AsyncClient", _Client)
    server = build_server(
        sample_num="1",
        images=[],
        cache_dir=tmp_path,
        save_dir=tmp_path / "save",
        controller="http://127.0.0.1:20001",
    )
    result = _call(server, "Calculator", {"expression": "2+2"})
    assert result.isError is False
    assert result.content[0].text == "4"
    assert seen[0][0] == "http://127.0.0.1:20001/get_worker_address"
    assert seen[0][1] == {"model": "Calculator"}
    assert seen[1][0] == "http://worker.test/worker_generate"
    assert seen[1][1]["expression"] == "2+2"


def test_call_worker_reports_a_missing_address():
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"address": ""}

    class _Client:
        async def post(self, url, json=None, timeout=None):
            del url, json, timeout
            return _Response()

    async def run():
        return await call_worker(_Client(), "http://controller", {}, "OCR", {"image": "/a.jpg"})

    body = asyncio.run(run())
    assert body["error_code"] == 1
    assert "not found" in body["text"]
