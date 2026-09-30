"""Summary keys and table rows for the W&B comparison hook. No W&B calls."""

from types import SimpleNamespace

from inspect_ai._util.registry import registry_find

import agentic_evals._registry  # noqa: F401  — registers the hook
from agentic_evals.agents.trace_callback import TRACE_KEY
from agentic_evals.wandb_compare import (
    RESULTS_COLUMNS,
    SAMPLE_COLUMNS,
    results_rows,
    sample_rows,
    summary_metrics,
)


def _metric(name: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


def _score(name: str, mean: float, stderr: float | None = None) -> SimpleNamespace:
    metrics = {"mean": _metric("mean", mean)}
    if stderr is not None:
        metrics["stderr"] = _metric("stderr", stderr)
    return SimpleNamespace(name=name, metrics=metrics)


def test_compare_hook_is_registered() -> None:
    found = registry_find(
        lambda info: info.type == "hooks" and info.name.endswith("/wandb_compare")
    )
    assert len(found) == 1


def test_summary_keys_and_results_table() -> None:
    scores = [
        _score("tao", 0.5),
        _score("tio", 0.4),
        _score("tem", 0.2),
        _score("param_acc", 0.1),
        _score("qwen/qwen3.8-27b/match", 1.0, stderr=0.2),
        _score("llama/match", 0.0, stderr=0.1),
        _score("merged/match", 0.5, stderr=0.15),
        _score("qwen/qwen3.8-27b/length_diff", -0.25, stderr=0.05),
        _score("llama/length_diff", -0.25, stderr=0.05),
        _score("accuracy", 1.0),
    ]

    summary = summary_metrics(scores, "geobenchx")

    assert summary == {
        "geobenchx/tao": 0.5,
        "geobenchx/tio": 0.4,
        "geobenchx/tem": 0.2,
        "geobenchx/param_acc": 0.1,
        "geobenchx/match": 0.5,
        "geobenchx/length_diff": -0.25,
    }
    rows = results_rows(scores, "geobenchx")
    assert list(rows[0]) == list(RESULTS_COLUMNS)
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["tao"] == {"task": "geobenchx", "metric": "tao", "value": 0.5, "stderr": None}
    assert by_metric["merged/match"]["stderr"] == 0.15
    assert by_metric["accuracy"]["value"] == 1.0


def test_sample_trace_table() -> None:
    sample = SimpleNamespace(
        id="task-7",
        epoch=1,
        metadata={"kind": "solvable"},
        scores={
            "geobenchx_steps": SimpleNamespace(
                value={"tao": 1.0, "tio": 1.0, "tem": 0.5, "param_acc": 0.5}
            ),
            "geobenchx_scorer": SimpleNamespace(
                value={"qwen/qwen3.8-27b/match": 2, "qwen/qwen3.8-27b/length_diff": 0.0}
            ),
            "geobenchx_merged": SimpleNamespace(value={"merged/match": 2}),
        },
        store={TRACE_KEY: [{"name": "get_osm_data"}, {"name": "buffer"}]},
    )

    rows = sample_rows([sample])

    assert list(rows[0]) == list(SAMPLE_COLUMNS)
    assert rows[0]["sample_id"] == "task-7"
    assert rows[0]["kind"] == "solvable"
    assert rows[0]["tao"] == 1.0
    assert rows[0]["match"] == 2
    assert rows[0]["length_diff"] == 0.0
    assert rows[0]["n_tool_calls"] == 2
    assert rows[0]["tool_trace"] == "get_osm_data -> buffer"
