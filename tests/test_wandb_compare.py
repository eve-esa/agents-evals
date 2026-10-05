"""Summary keys and table rows for the W&B comparison hook. No W&B calls."""

import json
from types import SimpleNamespace

from inspect_ai._util.registry import registry_find

import agentic_evals._registry  # noqa: F401  — registers the hook
from agentic_evals.agents.trace_callback import TRACE_KEY
from agentic_evals.wandb_compare import (
    METRICS_COLUMNS,
    SAMPLE_COLUMNS,
    dashboard_aliases,
    model_run_name,
    grouped_metrics,
    _trace_dtype,
    _trace_value,
    example_trace_rows,
    raw_trace,
    render_transcript,
    results_rows,
    sample_rows,
    sample_table_columns,
    sample_target,
    samples_for_table,
    summary_metrics,
    trace_records,
    trace_rows,
    trace_table_columns,
)


def _metric(name: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


def _score(
    name: str,
    mean: float,
    stderr: float | None = None,
    *,
    scored_samples: int | None = None,
    extra: dict[str, float] | None = None,
) -> SimpleNamespace:
    metrics: dict[str, SimpleNamespace] = {"mean": _metric("mean", mean)}
    if stderr is not None:
        metrics["stderr"] = _metric("stderr", stderr)
    for label, value in (extra or {}).items():
        metrics[label] = SimpleNamespace(name=label, value=value, group="verdict_rates")
    return SimpleNamespace(name=name, metrics=metrics, scored_samples=scored_samples)


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
        _score(
            "qwen/qwen3.8-27b/match",
            1.0,
            stderr=0.2,
            scored_samples=5,
            extra={"success": 0.8, "partial": 0.0, "no_match": 0.2},
        ),
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
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b": 1.0,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/stderr": 0.2,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/success": 0.8,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/partial": 0.0,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/no_match": 0.2,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n": 5,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n_success": 4,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n_partial": 0,
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n_no_match": 1,
        "geobenchx/llm_as_judge/llama": 0.0,
        "geobenchx/llm_as_judge/llama/stderr": 0.1,
        "geobenchx/llm_as_judge/avg": 0.5,
        "geobenchx/llm_as_judge/n_judges": 2,
        "geobenchx/match": 0.5,
        "geobenchx/length_diff": -0.25,
    }
    aliases = dashboard_aliases(summary, "geobenchx")
    assert aliases["geobenchx/llm_as_judge_avg"] == 0.5
    assert aliases["geobenchx/llm_as_judge_qwen_qwen3.8-27b"] == 1.0
    assert "geobenchx/tao" not in aliases
    assert "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n" not in aliases
    assert model_run_name("openai-api/openrouter/qwen/qwen3.8-27b") == "qwen3.8-27b"
    assert model_run_name("openai-api/openrouter/qwen/qwen3.8-27b", "geobenchx") == "qwen3.8-27b_geobenchx"
    rows = results_rows(scores, "geobenchx")
    assert list(rows[0]) == list(METRICS_COLUMNS)
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["tao"] == {
        "task": "geobenchx",
        "metric": "tao",
        "value": 0.5,
        "stderr": None,
        "n": 0,
    }
    assert by_metric["merged/match"]["stderr"] == 0.15
    assert by_metric["accuracy"]["value"] == 1.0


def test_sample_trace_table() -> None:
    sample = SimpleNamespace(
        id="task-7",
        epoch=1,
        target="",
        metadata={
            "kind": "solvable",
            "reference_solution_description": "Locate Rome.",
            "reference_solutions": [[{"name": "get_osm_data", "arguments": {"place": "Rome"}}]],
        },
        scores={
            "geobenchx_steps": SimpleNamespace(
                value={"tao": 1.0, "tio": 1.0, "tem": 0.5, "param_acc": 0.5}
            ),
            "geobenchx_scorer": SimpleNamespace(
                value={"qwen/qwen3.8-27b/match": 2, "qwen/qwen3.8-27b/length_diff": 0.0},
                explanation=None,
            ),
            "geobenchx_merged": SimpleNamespace(value={"merged/match": 2}),
            "trajectory_judge": SimpleNamespace(
                value={"trajectory_accuracy": 1.0},
                explanation="the tools match the reference",
                metadata={"judge_model": "openai-api/openrouter/deepseek/deepseek-v3.2"},
            ),
        },
        messages=[
            SimpleNamespace(role="user", text="Where is Rome?", tool_calls=None),
            SimpleNamespace(
                role="assistant",
                text="",
                tool_calls=[SimpleNamespace(function="get_osm_data", arguments={"place": "Rome"})],
            ),
            SimpleNamespace(role="tool", function="get_osm_data", text="ok", tool_calls=None),
            SimpleNamespace(role="assistant", text="Rome is in Italy.", tool_calls=None),
        ],
        store={
            TRACE_KEY: [
                {"step": 1, "agent": "react", "name": "get_osm_data", "input": {"place": "Rome"}, "output": "ok", "error": False},
                {"step": 2, "agent": "react", "name": "buffer", "input": {"radius": 100}, "output": "polygon", "error": False},
            ]
        },
    )

    rows = sample_rows([sample])

    assert list(rows[0]) == [
        *SAMPLE_COLUMNS,
        "deepseek/deepseek-v3.2/match",
        "deepseek/deepseek-v3.2/output",
        "qwen/qwen3.8-27b/match",
        "qwen/qwen3.8-27b/output",
    ]
    assert rows[0]["qwen/qwen3.8-27b/match"] == 2
    assert rows[0]["qwen/qwen3.8-27b/output"] == ""
    assert rows[0]["deepseek/deepseek-v3.2/match"] == 1.0
    assert rows[0]["deepseek/deepseek-v3.2/output"] == "the tools match the reference"
    assert sample_table_columns(rows) == tuple(rows[0])
    assert rows[0]["sample_id"] == "task-7"
    assert rows[0]["question"] == "Where is Rome?"
    assert rows[0]["final_answer"] == "Rome is in Italy."
    assert "[call get_osm_data]" in rows[0]["transcript"]
    assert "parameters:" in rows[0]["transcript"]
    assert "[tool get_osm_data]\nok" in rows[0]["transcript"]
    assert "======== turn 1 | user ========" in rows[0]["transcript"]
    assert rows[0]["kind"] == "solvable"
    assert "get_osm_data" in render_transcript(sample)
    traces = example_trace_rows([sample])
    assert trace_table_columns(traces) == (
        *SAMPLE_COLUMNS,
        "target",
        "raw_trace",
        "readable_trace",
        "deepseek/deepseek-v3.2/match",
        "deepseek/deepseek-v3.2/output",
        "qwen/qwen3.8-27b/match",
        "qwen/qwen3.8-27b/output",
    )
    assert traces[0]["question"] == "Where is Rome?"
    assert traces[0]["target"] == sample_target(sample)
    assert "reference 1" in traces[0]["target"]
    assert "get_osm_data" in traces[0]["target"]
    assert traces[0]["readable_trace"] == rows[0]["transcript"]
    assert "======== turn 3 | tool | get_osm_data ========" in traces[0]["readable_trace"]
    parsed = json.loads(traces[0]["raw_trace"])
    assert parsed["tool_trace"][0]["name"] == "get_osm_data"
    assert parsed["turns"][1]["tool_calls"][0]["arguments"] == {"place": "Rome"}
    assert raw_trace(sample) == traces[0]["raw_trace"]
    assert _trace_dtype("readable_trace") is str
    assert _trace_dtype("n_tool_calls") is int
    assert _trace_dtype("tao") is float
    assert _trace_value("readable_trace", None) == ""
    assert _trace_value("tao", None) == 0.0
    assert rows[0]["tao"] == 1.0
    assert rows[0]["match"] == 2
    assert rows[0]["length_diff"] == 0.0
    assert rows[0]["n_tool_calls"] == 2
    assert rows[0]["tool_trace"] == "get_osm_data -> buffer"
    calls = trace_rows([sample])
    assert [row["name"] for row in calls] == ["get_osm_data", "buffer"]
    assert calls[0]["tool_step"] == 1
    assert calls[0]["input"] == '{"place": "Rome"}'
    assert calls[0]["error_message"] == ""
    assert calls[1]["output"] == "polygon"
    saved = trace_records([sample])
    assert saved[0]["sample_id"] == "task-7"
    assert saved[0]["question"] == "Where is Rome?"
    assert "reference 1" in saved[0]["target"]
    assert len(saved[0]["trace"]) == 2
    assert saved[0]["readable_trace"].startswith("======== turn 1 | user ========")
    answered = SimpleNamespace(target="B", metadata={"reference_solutions": [[{"name": "reject_task", "arguments": {}}]]})
    assert sample_target(answered) == "B"


def test_each_class_repeats_the_summary() -> None:
    solvable = SimpleNamespace(
        metadata={"kind": "solvable", "labels": ["Spatial operations"]},
        scores={
            "steps": SimpleNamespace(value={"tao": 1.0, "tio": 0.5, "tem": 0.0, "param_acc": 0.0}),
            "judge": SimpleNamespace(
                value={
                    "qwen/qwen3.8-27b/match": 2,
                    "qwen/qwen3.8-27b/length_diff": 0.0,
                    "merged/match": 2,
                }
            ),
        },
    )
    unsolvable = SimpleNamespace(
        metadata={"kind": "unsolvable", "labels": ["Spatial operations", "Heatmaps, Contour Lines"]},
        scores={
            "steps": SimpleNamespace(value={"tao": 0.0, "tio": 0.0, "tem": 0.0, "param_acc": 0.0}),
            "judge": SimpleNamespace(
                value={
                    "qwen/qwen3.8-27b/match": 0,
                    "qwen/qwen3.8-27b/length_diff": 1.0,
                    "merged/match": 0,
                }
            ),
        },
    )

    grouped = grouped_metrics([solvable, unsolvable], "geobenchx", ["kind", "labels"])

    assert grouped["geobenchx/solvable/n"] == 1
    assert grouped["geobenchx/solvable/tao"] == 1.0
    assert grouped["geobenchx/solvable/match"] == 2
    assert grouped["geobenchx/solvable/llm_as_judge/avg"] == 2
    assert grouped["geobenchx/solvable/llm_as_judge/n_judges"] == 1
    assert grouped["geobenchx/solvable/llm_as_judge/qwen_qwen3.8-27b/n_success"] == 1
    assert grouped["geobenchx/unsolvable/tao"] == 0.0
    assert grouped["geobenchx/unsolvable/llm_as_judge/qwen_qwen3.8-27b"] == 0
    assert grouped["geobenchx/unsolvable/llm_as_judge/qwen_qwen3.8-27b/n_no_match"] == 1
    assert grouped["geobenchx/unsolvable/length_diff"] == 1.0
    # A list field counts the sample under every label it carries.
    assert grouped["geobenchx/Spatial_operations/n"] == 2
    assert grouped["geobenchx/Heatmaps_Contour_Lines/n"] == 1
    assert grouped["geobenchx/Spatial_operations/tao"] == 0.5
    assert not any("/by/" in key for key in grouped)


def test_sample_end_records_fill_an_empty_task_log() -> None:
    collected = [SimpleNamespace(id="task-1")]
    assert samples_for_table(None, collected) == collected
    assert samples_for_table([], collected) == collected
    logged = [SimpleNamespace(id="from-log")]
    assert samples_for_table(logged, []) == logged
