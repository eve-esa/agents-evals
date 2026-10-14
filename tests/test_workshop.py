"""Workshop question set and the results table built from an Inspect log."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sagemaker"))

from agentic_evals.agents.langgraph_react import TERMINATION_KEY, langgraph_agent, react_graph
from agentic_evals.agents.trace_callback import TRACE_KEY
from agentic_evals.tasks.earth_bench.dataset import load_samples
from agentic_evals.tasks.earth_bench.setup import _parse_questions
import pytest

from workshop import WORKSHOP_QUESTIONS, BudgetExceeded, judge, leaderboard, results_df, run


def test_workshop_questions_are_ten_spectrum_tasks_with_gold_trajectories():
    assert len(WORKSHOP_QUESTIONS) == 10
    assert len(set(WORKSHOP_QUESTIONS)) == 10
    assert WORKSHOP_QUESTIONS[0] == 11
    samples = {s.id: s for s in load_samples("ap", "spectrum")}
    for question in WORKSHOP_QUESTIONS:
        sample = samples[question]
        calls = sample.metadata["gold_tool_calls"]
        assert sample.metadata["has_gold_trajectory"]
        assert calls[0]["name"] == "get_filelist"
        assert len(calls) == (4 if question == 8 else 3)
    assert _parse_questions(",".join(str(q) for q in WORKSHOP_QUESTIONS)) == list(WORKSHOP_QUESTIONS)


def test_results_df_uses_scorer_fields_trace_length_and_termination():
    sample = SimpleNamespace(
        id="11",
        scores={
            "earth_bench_scorer": SimpleNamespace(
                value={
                    "accuracy": 1.0,
                    "tao": 1.0,
                    "tio": 0.5,
                    "tem": 0.5,
                    "param_acc": 0.0,
                    "efficiency": 1.0,
                    "tool_errors": 2.0,
                }
            ),
            "trajectory_judge": SimpleNamespace(value={"trajectory_accuracy": 1.0}),
        },
        store={TRACE_KEY: [{}, {}, {}], TERMINATION_KEY: "completed"},
    )
    frame = results_df(SimpleNamespace(samples=[sample]))
    assert frame.index.tolist() == [11]
    assert frame.loc[11, "n_calls"] == 3
    assert frame.loc[11, "termination"] == "completed"
    assert frame.loc[11, "tem"] == 0.5
    assert frame.loc[11, "trajectory_judge_trajectory_accuracy"] == 1.0


def test_langgraph_agent_accepts_a_graph_factory():
    agent = langgraph_agent(react_graph, recursion_limit=50, name="workshop_react")
    closed = [cell.cell_contents for cell in (agent.__closure__ or ())]
    assert react_graph in closed
    assert 50 in closed
    assert "workshop_react" in closed


def test_run_retries_a_silent_model_attempt(monkeypatch):
    captured: dict = {}

    def fake_eval(*_args, **kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(status="success", samples=[])]

    monkeypatch.setattr("inspect_ai.eval", fake_eval)
    monkeypatch.setattr("workshop.check_budget", lambda *_a, **_k: None)
    log = run(object(), model="openai-api/eve/x", questions=[11], log_dir="logs/unused")
    assert log.status == "success"
    assert captured["max_tokens"] == 4096
    assert captured["attempt_timeout"] == 180
    assert captured["stream_idle_timeout"] == 60
    assert captured["max_retries"] == 6
    assert captured["max_connections"] == 2
    assert captured["retry_on_error"] == 2


def _log(name, regime, ids, accuracy, status="success"):
    samples = [
        SimpleNamespace(
            id=str(q),
            scores={"earth_bench_scorer": SimpleNamespace(value={"accuracy": accuracy, "tao": 1.0, "tio": 1.0, "tem": 0.5, "param_acc": 0.5, "efficiency": 1.0, "tool_errors": 0.0})},
            store={TRACE_KEY: [{}], TERMINATION_KEY: "completed"},
        )
        for q in ids
    ]
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(task_registry_name="agentic_evals/earth_bench", model="openai-api/eve/x", task_args={"regime": regime}, solver=None),
        plan=SimpleNamespace(steps=[SimpleNamespace(params={"name": name})]),
        samples=samples,
    )


def test_leaderboard_keeps_finished_runs_on_the_workshop_questions(monkeypatch):
    logs = {
        "a.eval": _log("baseline", "ap", WORKSHOP_QUESTIONS, 0.5),
        "b.eval": _log("mine", "ap", WORKSHOP_QUESTIONS, 0.8),
        "c.eval": _log("partial", "ap", WORKSHOP_QUESTIONS[:3], 1.0),
        "d.eval": _log("crashed", "ap", WORKSHOP_QUESTIONS, 1.0, status="error"),
        "e.eval": _log("unscored", "ap", WORKSHOP_QUESTIONS, 1.0),
    }
    logs["e.eval"].samples[0].scores = {}
    monkeypatch.setattr("inspect_ai.log.list_eval_logs", lambda _dir: [SimpleNamespace(name=n) for n in logs])
    monkeypatch.setattr("inspect_ai.log.read_eval_log", lambda info: logs[info.name])
    board = leaderboard("logs")
    assert board["agent"].tolist() == ["mine", "baseline"]
    assert board.loc[0, "accuracy"] == 0.8
    assert board.loc[0, "n"] == 10
    assert "score" not in board.columns


def test_judge_appends_a_trajectory_judge_without_rerunning(monkeypatch):
    captured: dict = {}

    def fake_score(log, scorers, **kwargs):
        captured.update(kwargs, scorers=scorers)
        return log

    monkeypatch.setattr("inspect_ai.score", fake_score)
    log = object()
    assert judge(log, "openai-api/eve/x") is log
    assert captured["action"] == "append"
    assert len(captured["scorers"]) == 1


EVE_QUOTA = (
    "Error code: 429 - {'detail': \"Token budget exceeded for group 'eve_free'. Limit is 500000 tokens per period. "
    "Resets at 2026-11-05T08:42:59.676000+00:00.\", 'error': {'type': 'insufficient_quota', 'code': 'token_budget_exceeded'}}"
)


def test_run_stops_with_one_line_when_the_budget_runs_out(monkeypatch):
    log = SimpleNamespace(status="error", error=SimpleNamespace(message=EVE_QUOTA), samples=[])
    monkeypatch.setattr("inspect_ai.eval", lambda *_a, **_k: [log])
    monkeypatch.setattr("workshop.check_budget", lambda *_a, **_k: None)
    with pytest.raises(BudgetExceeded) as raised:
        run(object(), model="openai-api/eve/x", questions=[11], log_dir="logs/unused")
    assert str(raised.value).endswith("Resets at 2026-11-05T08:42:59.676000+00:00.")
    assert "'eve_free'" in str(raised.value)


def test_check_budget_raises_on_a_spent_quota(monkeypatch):
    import workshop

    response = SimpleNamespace(status_code=429, text=EVE_QUOTA)
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: response)
    monkeypatch.setattr(workshop, "eve_credentials", lambda: ("m", "https://example.invalid/v1", "k"))
    with pytest.raises(BudgetExceeded):
        workshop.check_budget()
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: SimpleNamespace(status_code=200, text="{}"))
    workshop.check_budget()


def test_compare_html_marks_each_step_and_reports_the_scores():
    from workshop import compare_html

    def call(name, **kw):
        return {"name": name, "input": kw}

    gold = [call("get_filelist", dir_path="d"), call("lst", out="a.tif"), call("ratio", threshold=310.0)]
    pred = [call("get_filelist", dir_path="d"), call("band_ratio", x=1), call("ratio", threshold=300.0), call("extra")]
    page = compare_html(pred, gold, "run e")
    # one status icon per row: match, different tool, different arguments (threshold 300, not 310), extra call
    for icon in ("\u2713", "\u2717", "\u2260", "+"):
        assert f"justify-content:center'>{icon}</div>" in page
    assert page.count("justify-content:center'>") == 4
    assert "0.33" in page and "agent vs 3 expert" in page
    assert "(expert: 310.0)" in page
