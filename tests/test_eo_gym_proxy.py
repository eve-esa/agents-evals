"""The EO-Gym MCP proxy against a stub ``/execute`` server (offline)."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from agentic_evals.paths import EO_GYM_TOOLS_JSON
from agentic_evals.tasks.eo_gym.tools.mcp_proxy import execute_output, new_image_ref
from agentic_evals.tasks.eo_gym.tools import proxy_servers

CALLS: list[dict] = []


class StubExecute(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        CALLS.append(payload)
        body = {"output": {"count": 3}} if payload["tool_name"] == "basic_calculator" else {"output": "Error: bad input"}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_execute_output_mirrors_driver():
    assert execute_output(httpx.Response(200, text='{"output": "ok"}')) == "ok"
    assert execute_output(httpx.Response(200, text='{"output": {"a": 1}}')) == '{"a": 1}'
    assert execute_output(httpx.Response(500, text="boom")).startswith("Error: EO Tool API returned HTTP 500")
    assert new_image_ref("crop_optical_or_sar_image", '{"cropped_path": "EO_GYM_DATA/x.png"}') == "EO_GYM_DATA/x.png"
    assert new_image_ref("basic_calculator", '{"cropped_path": "x"}') is None


@pytest.mark.skipif(not EO_GYM_TOOLS_JSON.exists(), reason="tool schemas not exported")
def test_proxy_round_trip():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    httpd = HTTPServer(("127.0.0.1", 0), StubExecute)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/execute"

    async def run():
        client = MultiServerMCPClient(proxy_servers(["basic_calculator", "theme_index_lookup"], "dior", url))
        tools = {t.name: t for t in await client.get_tools()}
        ok = await tools["basic_calculator"].ainvoke({"expression": "1+2"})
        err = await tools["theme_index_lookup"].ainvoke({"detections": "[]"})
        return tools, ok, err

    try:
        tools, ok, err = asyncio.run(run())
    finally:
        httpd.shutdown()
    assert set(tools) == {"basic_calculator", "theme_index_lookup"}
    assert json.loads(ok[0]["text"]) == {"count": 3}
    assert "Error: bad input" in str(err)
    assert CALLS[0] == {"tool_name": "basic_calculator", "arguments": {"expression": "1+2"}}
