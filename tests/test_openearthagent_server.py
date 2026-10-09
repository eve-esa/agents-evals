"""Offline checks for the OpenEarthAgent server launcher. No conda, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_evals.tasks.openearthagent import tools as tools_mod
from agentic_evals.tasks.openearthagent.tools import server

EXAMPLE = """\
base_dir: OpenEarthAgent/tool_server/tool_workers
save_path: OpenEarthAgent/tools_output
services:
  - worker_name: controller
  - worker_name: OCR
    cuda_visible_devices: "3"
  - worker_name: ObjectDetection
    cuda_visible_devices: "0"
    model-path: OpenEarthAgent/models/groundingdino_swint_ogc.pth
  - worker_name: GoogleSearch
    api-key: "<SERPER_API_KEY>"
  - worker_name: AddIndexLayer
    service_account: "<service_account>"
    key_file: gee_key/<key_file>
"""


def test_render_uses_absolute_paths_default_gpu_and_env_secrets(tmp_path: Path):
    clone = tmp_path / "upstream"
    example_path = clone / server.EXAMPLE_REL
    example_path.parent.mkdir(parents=True)
    example_path.write_text(EXAMPLE, encoding="utf-8")
    gee = tmp_path / "gee.json"
    gee.write_text(json.dumps({"client_email": "svc@example.iam.gserviceaccount.com"}), encoding="utf-8")
    rendered = server.render_server_config(
        clone,
        environ={
            "OPENEARTHAGENT_SERPER_API_KEY": "serper-test-key",
            "OPENEARTHAGENT_GEE_KEY": str(gee),
        },
    )
    root = clone.resolve().as_posix()
    assert f"base_dir: {root}/tool_server/tool_workers" in rendered
    assert f"save_path: {root}/tools_output" in rendered
    assert f"model-path: {root}/models/groundingdino_swint_ogc.pth" in rendered
    assert "OpenEarthAgent/" not in rendered
    assert 'cuda_visible_devices: "0"' in rendered
    assert 'cuda_visible_devices: "3"' not in rendered
    assert 'api-key: "serper-test-key"' in rendered
    assert 'service_account: "svc@example.iam.gserviceaccount.com"' in rendered
    assert f"key_file: {gee.resolve().as_posix()}" in rendered
    assert server.expected_workers(rendered) == ["OCR", "ObjectDetection", "GoogleSearch", "AddIndexLayer"]


def test_missing_secret_names_the_variable(tmp_path: Path):
    clone = tmp_path / "upstream"
    example_path = clone / server.EXAMPLE_REL
    example_path.parent.mkdir(parents=True)
    example_path.write_text(EXAMPLE, encoding="utf-8")
    with pytest.raises(RuntimeError, match="OPENEARTHAGENT_SERPER_API_KEY"):
        server.render_server_config(clone, environ={})


def test_existing_controller_is_left_running(monkeypatch: pytest.MonkeyPatch):
    tools_mod._PROBE.clear()
    monkeypatch.setattr(server, "_listed", lambda url: ["OCR"])
    monkeypatch.setattr(server, "_OWNED", False)

    def boom(url: str, clone: Path) -> dict:
        raise AssertionError("launcher called")

    monkeypatch.setattr(server, "_start", boom)
    record = server.ensure_controller("http://127.0.0.1:20001")
    assert record["started_by_eval"] is False
    assert record["models"] == ["OCR"]


def test_remote_controller_is_not_started_locally(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_listed", lambda url: None)

    def boom(url: str, clone: Path) -> dict:
        raise AssertionError("launcher called")

    monkeypatch.setattr(server, "_start", boom)
    with pytest.raises(RuntimeError, match="remote"):
        server.ensure_controller("http://example.com:20001")


def test_missing_clone_is_named_before_conda(tmp_path: Path):
    with pytest.raises(RuntimeError, match="start_server_local.py"):
        server._start("http://127.0.0.1:20001", tmp_path)
