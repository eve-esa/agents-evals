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


def test_a_specialists_sample_nests_each_specialist_run_under_its_a2a_task(monkeypatch):
    from datetime import datetime, timezone

    from agentic_evals.agents.trace_callback import RAW_TRACE_KEY, SUBAGENT_TRACES_KEY

    def at(t):
        return datetime.fromtimestamp(t, tz=timezone.utc)

    def model(system, start, end):
        return SimpleNamespace(
            event="model", input=[SimpleNamespace(content=system)], timestamp=at(start), completed=at(end),
            working_time=end - start,
        )

    ask = SimpleNamespace(function="ask_thermal", arguments={"request": "LST of a.tif"})
    messages = [
        SimpleNamespace(role="user", text="Instructions.\nQuestion: How hot?", tool_calls=None),
        SimpleNamespace(role="assistant", text="", tool_calls=[ask]),
        SimpleNamespace(role="tool", text="LST is 312 K", function="ask_thermal", tool_calls=None),
        SimpleNamespace(role="assistant", text="<Answer>C<Answer>", tool_calls=None),
    ]
    # The specialist's calls (system prompt "spec") interleave with the orchestrator's ("orch").
    events = [model("orch", 0, 2), model("spec", 3, 4), model("spec", 6, 7), model("orch", 9, 10)]
    artifact = {"agent": "thermal", "task_id": "t-1", "context_id": "c-1", "state": "completed"}
    store = {
        RAW_TRACE_KEY: [{"name": "ask_thermal", "input": {"request": "LST of a.tif"}, "started_at": 2.5,
                         "ended_at": 8.0, "artifact": artifact}],
        SUBAGENT_TRACES_KEY: {"thermal": {
            "name": "Thermal Agent",
            "context_ids": ["c-1"],
            "messages": [
                {"role": "human", "text": "LST of a.tif"},
                {"role": "ai", "text": "", "tool_calls": [{"name": "lst", "args": {"path": "a.tif"}}]},
                {"role": "tool", "text": "312", "name": "lst"},
                {"role": "ai", "text": "LST is 312 K"},
            ],
            "calls": [{"name": "lst", "input": {"path": "a.tif"}, "started_at": 4.5, "ended_at": 5.5}],
        }},
    }
    result = trace_from_sample(SimpleNamespace(messages=messages, events=events, store=store))

    orchestrator, delegation, final = result["trace"]
    assert (orchestrator["latency_s"], final["latency_s"]) == (2, 1)   # the specialist's calls are not the orchestrator's
    assert delegation["node"] == "a2a" and delegation["a2a"] == artifact
    assert delegation["request"] == "LST of a.tif" and delegation["latency_s"] == 5.5
    assert [(s["node"], s["latency_s"]) for s in delegation["subtrace"]] == [("agent", 1), ("tools", 1.0), ("agent", 1)]
    assert [lane["agent"] for lane in result["lanes"]] == ["Orchestrator", "Thermal Agent"]

    captured = []
    monkeypatch.setattr("trace_view.display", lambda obj: captured.append(obj.data))
    visualize_agent_trace(result)
    page = captured[0]
    assert "A2A Delegation" in page and "Thermal Agent step" in page and "Step 2.2" in page
    assert "one row per agent" in page
