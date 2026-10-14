"""The workshop trace view turns an Inspect sample into agent and tool steps."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("IPython", reason="the workshop trace view needs IPython (sagemaker/.venv)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sagemaker"))

from trace_view import trace_from_sample, visualize_agent_trace  # noqa: E402


def test_trace_from_sample_maps_messages_to_agent_and_tool_steps(monkeypatch):
    call = SimpleNamespace(function="get_filelist", arguments={"dir_path": "d"})
    messages = [
        SimpleNamespace(role="user", text="Instructions.\nQuestion: How much? benchmark/data/question11", tool_calls=None),
        SimpleNamespace(role="assistant", text="", tool_calls=[call]),
        SimpleNamespace(role="tool", text='["a.tif"]', function="get_filelist", tool_calls=None),
        SimpleNamespace(role="assistant", text="<Answer>C<Answer>", tool_calls=None),
    ]
    events = [SimpleNamespace(event="model", working_time=2.5), SimpleNamespace(event="model", working_time=1.0)]
    result = trace_from_sample(SimpleNamespace(messages=messages, events=events))

    assert result["query"].startswith("How much?")
    assert [s["node"] for s in result["trace"]] == ["agent", "tools", "agent"]
    assert result["trace"][0]["tool_calls"] == [{"name": "get_filelist", "args": {"dir_path": "d"}}]
    assert result["trace"][0]["latency_s"] == 2.5
    assert result["trace"][1]["name"] == "get_filelist" and "a.tif" in result["trace"][1]["content"]

    captured = []
    monkeypatch.setattr("trace_view.display", lambda obj: captured.append(obj.data))
    visualize_agent_trace(result)
    page = captured[0]
    assert "Agent execution trace" in page and "Tool Execution" in page
    assert "Timeline" not in page          # no tool timings, so no timeline bar
