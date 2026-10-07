"""Summary keys and table rows for the W&B comparison hook. No W&B calls."""

import json
from types import SimpleNamespace

from inspect_ai._util.registry import registry_find

import agentic_evals._registry  # noqa: F401  — registers the hook
from agentic_evals.agents.trace_callback import TRACE_KEY
from agentic_evals.wandb_compare import (
    METRICS_COLUMNS,
    SAMPLE_COLUMNS,
    comparison_summary,
    dashboard_aliases,
    metric_root,
    model_run_name,
    grouped_metrics,
    native_metrics,
    published_names,
    _trace_dtype,
    _trace_value,
    example_trace_rows,
    geobenchx_share_summary,
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
        _score("qwen/qwen3.8-27b/match", 0.8, stderr=0.2, scored_samples=5),
        _score("qwen/qwen3.8-27b/partial", 0.0, scored_samples=5),
        _score("qwen/qwen3.8-27b/no_match", 0.2, scored_samples=5),
        _score("llama/match", 0.0, stderr=0.1),
        _score("llama/partial", 0.0),
        _score("llama/no_match", 1.0),
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
        "geobenchx/llm_as_judge/qwen_qwen3.8-27b": 0.8,
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
        "geobenchx/llm_as_judge/llama/success": 0.0,
        "geobenchx/llm_as_judge/llama/partial": 0.0,
        "geobenchx/llm_as_judge/llama/no_match": 1.0,
        "geobenchx/llm_as_judge/avg": 0.4,
        "geobenchx/llm_as_judge/n_judges": 2,
        "geobenchx/match": 0.4,
        "geobenchx/partial": 0.0,
        "geobenchx/no_match": 0.6,
        "geobenchx/length_diff": -0.25,
    }
    aliases = dashboard_aliases(summary, "geobenchx")
    assert aliases["geobenchx/llm_as_judge_avg"] == 0.4
    assert aliases["geobenchx/llm_as_judge_qwen_qwen3.8-27b"] == 0.8
    assert "geobenchx/tao" not in aliases
    assert "geobenchx/llm_as_judge/qwen_qwen3.8-27b/n" not in aliases
    assert model_run_name("openai-api/openrouter/qwen/qwen3.8-27b") == "qwen3.8-27b"
    assert (
        model_run_name("openai-api/openrouter/qwen/qwen3.8-27b", "geobenchx")
        == "qwen3.8-27b_geobenchx"
    )
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
            "reference_solutions": [
                [{"name": "get_osm_data", "arguments": {"place": "Rome"}}]
            ],
        },
        scores={
            "geobenchx_steps": SimpleNamespace(
                value={"tao": 1.0, "tio": 1.0, "tem": 0.5, "param_acc": 0.5}
            ),
            "geobenchx_scorer": SimpleNamespace(
                value={
                    "qwen/qwen3.8-27b/match": 2,
                    "qwen/qwen3.8-27b/length_diff": 0.0,
                },
                explanation=None,
            ),
            "geobenchx_merged": SimpleNamespace(value={"merged/match": 2}),
            "trajectory_judge": SimpleNamespace(
                value={"trajectory_accuracy": 1.0},
                explanation="the tools match the reference",
                metadata={
                    "judge_model": "openai-api/openrouter/deepseek/deepseek-v3.2"
                },
            ),
        },
        messages=[
            SimpleNamespace(role="user", text="Where is Rome?", tool_calls=None),
            SimpleNamespace(
                role="assistant",
                text="",
                tool_calls=[
                    SimpleNamespace(
                        function="get_osm_data", arguments={"place": "Rome"}
                    )
                ],
            ),
            SimpleNamespace(
                role="tool", function="get_osm_data", text="ok", tool_calls=None
            ),
            SimpleNamespace(
                role="assistant", text="Rome is in Italy.", tool_calls=None
            ),
        ],
        store={
            TRACE_KEY: [
                {
                    "step": 1,
                    "agent": "react",
                    "name": "get_osm_data",
                    "input": {"place": "Rome"},
                    "output": "ok",
                    "error": False,
                },
                {
                    "step": 2,
                    "agent": "react",
                    "name": "buffer",
                    "input": {"radius": 100},
                    "output": "polygon",
                    "error": False,
                },
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
    assert (
        "======== turn 3 | tool | get_osm_data ========" in traces[0]["readable_trace"]
    )
    parsed = json.loads(traces[0]["raw_trace"])
    assert parsed["tool_trace"][0]["name"] == "get_osm_data"
    assert parsed["turns"][1]["tool_calls"][0]["arguments"] == {"place": "Rome"}
    assert raw_trace(sample) == traces[0]["raw_trace"]
    assert _trace_dtype("readable_trace") is str
    assert _trace_dtype("n_tool_calls") is int
    assert _trace_dtype("tao") is float
    assert _trace_value("readable_trace", None) == ""
    assert _trace_value("tao", None) is None
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
    answered = SimpleNamespace(
        target="B",
        metadata={"reference_solutions": [[{"name": "reject_task", "arguments": {}}]]},
    )
    assert sample_target(answered) == "B"


def test_each_class_repeats_the_summary() -> None:
    solvable = SimpleNamespace(
        metadata={"kind": "solvable", "labels": ["Spatial operations"]},
        scores={
            "steps": SimpleNamespace(
                value={"tao": 1.0, "tio": 0.5, "tem": 0.0, "param_acc": 0.0}
            ),
            "judge": SimpleNamespace(
                value={
                    "qwen/qwen3.8-27b/match": 1.0,
                    "qwen/qwen3.8-27b/partial": 0.0,
                    "qwen/qwen3.8-27b/no_match": 0.0,
                    "qwen/qwen3.8-27b/length_diff": 0.0,
                    "merged/match": 1.0,
                }
            ),
        },
    )
    unsolvable = SimpleNamespace(
        metadata={
            "kind": "unsolvable",
            "labels": ["Spatial operations", "Heatmaps, Contour Lines"],
        },
        scores={
            "steps": SimpleNamespace(
                value={"tao": 0.0, "tio": 0.0, "tem": 0.0, "param_acc": 0.0}
            ),
            "judge": SimpleNamespace(
                value={
                    "qwen/qwen3.8-27b/match": 0.0,
                    "qwen/qwen3.8-27b/partial": 0.0,
                    "qwen/qwen3.8-27b/no_match": 1.0,
                    "qwen/qwen3.8-27b/length_diff": 1.0,
                    "merged/match": 0.0,
                }
            ),
        },
    )

    grouped = grouped_metrics([solvable, unsolvable], "geobenchx", ["kind", "labels"])

    assert grouped["geobenchx/solvable/n"] == 1
    assert grouped["geobenchx/solvable/tao"] == 1.0
    assert grouped["geobenchx/solvable/match"] == 1.0
    assert grouped["geobenchx/solvable/llm_as_judge/avg"] == 1.0
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


def test_panel_share_weights_each_judge_once_when_one_skips_a_task() -> None:
    # A scored two tasks (match share 0.5). B scored only the first (share 1).
    # The per-task average of indicators is 0.5; Section 5.2 is 0.75.
    scores = [
        _score("a/match", 0.5, scored_samples=2),
        _score("a/no_match", 0.5, scored_samples=2),
        _score("b/match", 1.0, scored_samples=1),
        _score("b/no_match", 0.0, scored_samples=1),
        _score("merged/match", 0.5),
    ]
    summary = summary_metrics(scores, "geobenchx")
    assert summary["geobenchx/match"] == 0.75
    assert summary["geobenchx/no_match"] == 0.25
    assert summary["geobenchx/llm_as_judge/avg"] == 0.75
    assert summary["geobenchx/llm_as_judge/n_judges"] == 2

    first = SimpleNamespace(
        metadata={"kind": "solvable"},
        scores={
            "j": SimpleNamespace(
                value={
                    "a/match": 1.0,
                    "a/partial": 0.0,
                    "a/no_match": 0.0,
                    "b/match": 1.0,
                    "b/partial": 0.0,
                    "b/no_match": 0.0,
                    "merged/match": 1.0,
                }
            )
        },
    )
    second = SimpleNamespace(
        metadata={"kind": "solvable"},
        scores={
            "j": SimpleNamespace(
                value={
                    "a/match": 0.0,
                    "a/partial": 0.0,
                    "a/no_match": 1.0,
                    "merged/match": 0.0,
                }
            )
        },
    )
    grouped = grouped_metrics([first, second], "geobenchx", ["kind"])
    assert grouped["geobenchx/solvable/match"] == 0.75
    assert grouped["geobenchx/solvable_incl_control/match"] == 0.75

    kept = geobenchx_share_summary([first, second])
    assert kept["geobenchx/match"] == 0.75
    assert kept["geobenchx/solvable_incl_control/n"] == 2
    assert kept["geobenchx/llm_as_judge_avg"] == 0.75
    assert not any(key.endswith("/median") or key.endswith("/tao") for key in kept)


def _panel_score(model: str, tool: float, answer: float | None) -> SimpleNamespace:
    value = {"inst": 1.0, "tool": tool, "argn": 1.0, "argv": 0.0}
    if answer is not None:
        value["answer"] = answer
    return SimpleNamespace(value=value, metadata={"judge_model": model})


def test_openearthagent_trace_table_uses_its_own_metrics() -> None:
    judges = {
        "deepseek-v3.2": "openai-api/openrouter/deepseek/deepseek-v3.2",
        "gpt-6-luna": "openai-api/openrouter/openai/gpt-6-luna",
    }
    sample = SimpleNamespace(
        id="0:6",
        epoch=1,
        target="",
        metadata={"regime": "step", "kind": ""},
        scores={
            leaf: _panel_score(model, 1.0, answer)
            for (leaf, model), answer in zip(judges.items(), (-1.0, 0.5), strict=True)
        },
        messages=[
            SimpleNamespace(role="user", text="Where is the port?", tool_calls=None)
        ],
        store={},
    )

    rows = example_trace_rows([sample])
    columns = trace_table_columns(rows)

    assert "tao" not in columns
    assert "match" not in columns
    assert "inst" in columns
    assert "tool" in columns
    assert "argn" in columns
    assert "argv" in columns
    assert "deepseek-v3.2/summ" in columns
    assert "gpt-6-luna/summ" in columns
    assert rows[0]["tool"] == 1.0
    assert rows[0]["deepseek-v3.2/summ"] == -1.0
    assert rows[0]["gpt-6-luna/summ"] == 0.5
    assert rows[0]["n_tool_calls"] == 0
    assert "sample_id" in columns and "readable_trace" in columns


def test_openearthagent_panel_keeps_one_tool_score_and_four_summ_keys() -> None:
    judges = {
        "deepseek-v3.2": "openai-api/openrouter/deepseek/deepseek-v3.2",
        "gpt-6-luna": "openai-api/openrouter/openai/gpt-6-luna",
        "qwen3.8-flash": "openai-api/openrouter/qwen/qwen3.8-flash",
        "mistral-large-2512": "openai-api/mistral/mistral-large-2512",
    }
    # The second sample has no answer, so that NaN stays out of each Summ mean.
    answers = (
        {
            "deepseek-v3.2": -1.0,
            "gpt-6-luna": 1.0,
            "qwen3.8-flash": 0.5,
            "mistral-large-2512": 1.0,
        },
        {
            "deepseek-v3.2": None,
            "gpt-6-luna": 0.99,
            "qwen3.8-flash": 1.0,
            "mistral-large-2512": 0.975,
        },
    )
    samples = [
        SimpleNamespace(
            metadata={"regime": "step", "input_kind": "text", "task_kind": "answer"},
            scores={
                f"judge-{leaf}": _panel_score(model, 1.0, row[leaf])
                for leaf, model in judges.items()
            },
        )
        for row in answers
    ]

    summary = native_metrics(samples, "openearthagent", regime="step")

    assert summary["openearthagent/tool"] == 1.0
    assert summary["openearthagent/inst"] == 1.0
    assert summary["openearthagent/argn"] == 1.0
    assert summary["openearthagent/argv"] == 0.0
    assert "openearthagent/summ" not in summary
    assert "openearthagent/answer" not in summary
    assert summary["openearthagent/deepseek-v3.2/summ"] == -1.0
    assert summary["openearthagent/gpt-6-luna/summ"] == 0.995
    assert summary["openearthagent/qwen3.8-flash/summ"] == 0.75
    assert summary["openearthagent/mistral-large-2512/summ"] == 0.9875
    assert not any(key.endswith("/tool") and key.count("/") > 1 for key in summary)

    grouped = grouped_metrics(samples, "openearthagent", ["input_kind"], regime="step")
    assert grouped["openearthagent/text/tool"] == 1.0
    assert grouped["openearthagent/text/argv"] == 0.0
    assert "openearthagent/text/summ" not in grouped
    assert grouped["openearthagent/text/deepseek-v3.2/summ"] == -1.0
    assert grouped["openearthagent/text/gpt-6-luna/summ"] == 0.995
    assert grouped["openearthagent/text/qwen3.8-flash/summ"] == 0.75
    assert grouped["openearthagent/text/mistral-large-2512/summ"] == 0.9875


def test_openearthagent_e2e_answer_is_ans_and_f1_is_shared() -> None:
    sample = SimpleNamespace(
        metadata={"regime": "e2e"},
        scores={
            "openearthagent_scorer": _panel_score(
                "openai-api/openrouter/deepseek/deepseek-v3.2", 1.0, 0.2
            ),
            "openearthagent_scorer1": _panel_score(
                "openai-api/openrouter/openai/gpt-6-luna", 1.0, 0.4
            ),
        },
    )
    sample.scores["openearthagent_scorer"].value["any_order"] = 1.0
    sample.scores["openearthagent_scorer1"].value["any_order"] = 1.0
    f1 = {
        "perception": SimpleNamespace(
            name="perception", group="category_f1", value=0.5
        ),
        "operation": SimpleNamespace(name="operation", group="category_f1", value=0.25),
    }
    scores = [
        SimpleNamespace(
            scorer="openearthagent_scorer", name="openearthagent_scorer", metrics=f1
        ),
        SimpleNamespace(
            scorer="openearthagent_scorer1", name="openearthagent_scorer1", metrics=f1
        ),
    ]

    summary = native_metrics([sample], "openearthagent", regime="e2e", scores=scores)

    assert summary["openearthagent/any_order"] == 1.0
    assert summary["openearthagent/deepseek-v3.2/ans"] == 0.2
    assert summary["openearthagent/gpt-6-luna/ans"] == 0.4
    assert "openearthagent/summ" not in summary
    assert summary["openearthagent/perception"] == 0.5
    assert summary["openearthagent/operation"] == 0.25
    assert not any("/perception" in key and key.count("/") > 1 for key in summary)


def test_earth_bench_charts_are_the_paper_metrics_under_the_regime() -> None:
    # Feng et al., Section 4 and Appendix B.2: AP and IF are separate, and the
    # chart is accuracy, efficiency, and the four step scores. Modality is a
    # section under the regime. accuracy_other_key and trajectory judges are not.
    only = published_names(
        {
            "published_metrics": [
                "accuracy",
                "efficiency",
                "tao",
                "tio",
                "tem",
                "param_acc",
            ]
        }
    )
    assert metric_root("earth_bench", "ap") == "earth_bench/AP"
    assert metric_root("earth_bench", "if") == "earth_bench/IF"
    assert metric_root("openearthagent", "step") == "openearthagent"
    sample = SimpleNamespace(
        metadata={"modality": "spectrum", "regime": "ap"},
        scores={
            "earth_bench_scorer": SimpleNamespace(
                value={
                    "accuracy": 1.0,
                    "accuracy_other_key": 0.0,
                    "efficiency": 1.5,
                    "tao": 0.5,
                    "tio": 0.25,
                    "tem": 0.0,
                    "param_acc": 0.0,
                    "tool_errors": 3.0,
                    "perception_failures": 1.0,
                }
            ),
            "trajectory_judge": SimpleNamespace(
                value={"trajectory_accuracy": 1.0},
                metadata={
                    "judge_model": "openai-api/openrouter/deepseek/deepseek-v3.2"
                },
            ),
        },
    )
    summary = comparison_summary(
        [sample],
        [],
        "earth_bench",
        regime="ap",
        groups=["modality"],
        only=only,
    )
    assert summary["earth_bench/AP/accuracy"] == 1.0
    assert summary["earth_bench/AP/efficiency"] == 1.5
    assert summary["earth_bench/AP/tao"] == 0.5
    assert summary["earth_bench/AP/spectrum/accuracy"] == 1.0
    assert summary["earth_bench/AP/spectrum/tao"] == 0.5
    assert not any(
        "accuracy_other_key" in key or "trajectory_accuracy" in key for key in summary
    )
    assert not any(
        key.endswith("/tool_errors") or key.endswith("/perception_failures")
        for key in summary
    )


def test_sample_end_records_fill_an_empty_task_log() -> None:
    collected = [SimpleNamespace(id="task-1")]
    assert samples_for_table(None, collected) == collected
    assert samples_for_table([], collected) == collected
    logged = [SimpleNamespace(id="from-log")]
    assert samples_for_table(logged, []) == logged
