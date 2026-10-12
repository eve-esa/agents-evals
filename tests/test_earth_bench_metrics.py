"""Reproduce upstream's published Earth-Bench scores from their published predictions.

Our metric code, fed the
13 models x 2 regimes of published trajectories, must reproduce every per-task
score in ``step_by_step_evaluation_results.json`` and
``end_to_end_evaluation_results.json``. The published files cover 59 RGB tasks
(question190-question248); see ``metrics.upstream_scored_questions``.
"""

from __future__ import annotations

import json
import math

import pytest

from agentic_evals.tasks.earth_bench import metrics
from agentic_evals.tasks.earth_bench.tools import scalar_str_params
from conftest import published_run_dirs, read_json

pytestmark = pytest.mark.upstream

STEP_METRICS = {
    "contains_all_tool_calls_any_order": "tao",
    "contains_all_tool_calls_in_order": "tio",
    "trajectory_step_wise_score": "tem",
    "parameter_accuracy": "param_acc",
}


def _qkey(raw: str) -> str:
    return f"question{raw}" if raw.isdigit() else raw


def _published(earth_agent_root):
    dirs = published_run_dirs(earth_agent_root)
    assert len(dirs) == 26, [d.name for d in dirs]
    return dirs


def test_upstream_scored_questions_skip_question189(earth_bench_gold):
    ids = [g["question_index"] for g in earth_bench_gold]
    assert "question43" not in ids
    scored = metrics.upstream_scored_questions(ids)
    assert scored[0] == "question190" and scored[-1] == "question248" and len(scored) == 59


def test_step_by_step_matches_published(earth_agent_root, earth_bench_gold):
    gold = {g["question_index"]: g for g in earth_bench_gold}
    scope = metrics.upstream_scored_questions(list(gold))
    compared = 0
    for run_dir in _published(earth_agent_root):
        preds = {_qkey(p["question_index"]): p for p in read_json(run_dir / "extracted_tool_calls.json")}
        published = read_json(run_dir / "step_by_step_evaluation_results.json")
        individual = published["individual_results"]
        assert set(individual) == {q for q in scope if q in preds}, run_dir.name
        for q, expected in individual.items():
            ours = metrics.score_trajectory(preds[q]["tool_calls"], gold[q]["tool_calls"], upstream_compat=True).values()
            for upstream_key, our_key in STEP_METRICS.items():
                assert ours[our_key] == pytest.approx(expected[upstream_key]["score"]), (run_dir.name, q, our_key)
                compared += 1
        for upstream_key, stats in published["summary"]["metrics_summary"].items():
            mean = sum(
                metrics.score_trajectory(preds[q]["tool_calls"], gold[q]["tool_calls"], upstream_compat=True).values()[
                    STEP_METRICS[upstream_key]
                ]
                for q in individual
            ) / len(individual)
            assert mean == pytest.approx(stats["average_score"]), (run_dir.name, upstream_key)
    assert compared > 0


def test_end_to_end_matches_published(earth_agent_root, earth_bench_gold):
    gold = {g["question_index"]: g for g in earth_bench_gold}
    for run_dir in _published(earth_agent_root):
        answers = {
            _qkey(a["question_id"]): a
            for a in read_json(run_dir / "results_summary_polished.json")
        }
        calls = {_qkey(p["question_index"]): p for p in read_json(run_dir / "extracted_tool_calls.json")}
        published = read_json(run_dir / "end_to_end_evaluation_results.json")

        correct = 0
        evaluated = 0
        for row in published["accuracy"]["detailed_results"]:
            q = row["question_index"]
            if row["status"] == "missing":
                assert q not in answers, (run_dir.name, q)
                continue
            evaluated += 1
            answer = answers[q]
            text = answer.get("final_answer") or answer.get("polished_answer", "")
            extracted, status, is_correct = metrics.answer_status(text, gold[q]["final_answer"])
            assert (extracted, status, is_correct) == (row["predicted"], row["status"], row["correct"]), (run_dir.name, q)
            correct += is_correct
        assert evaluated == published["accuracy"]["evaluated_questions"]
        assert correct / evaluated == pytest.approx(published["accuracy"]["accuracy"]), run_dir.name

        effs = []
        for row in published["efficiency"]["detailed_results"]:
            q = row["question_index"]
            if row["status"] == "missing":
                continue
            ours = metrics.efficiency(calls[q]["tool_calls"], gold[q]["tool_calls"])
            assert ours == pytest.approx(row["efficiency"]), (run_dir.name, q)
            effs.append(ours)
        avg = published["efficiency"]["average_efficiency"]
        assert (sum(effs) / len(effs)) == pytest.approx(avg) or (math.isinf(avg) and math.isinf(sum(effs))), run_dir.name


def test_expand_scalar_gold_on_pinned_gt(earth_bench_gold):
    """question70's TVDI step is two parallel paths; question1's lists differ in length."""
    params = scalar_str_params()
    assert {"ndvi_path", "lst_path", "output_path"} <= params["compute_tvdi"]
    assert "file_list" not in params["calculate_tif_average"]
    assert "image_paths" not in params["calculate_threshold_ratio"]

    by_question = {entry["question_index"]: entry["tool_calls"] for entry in earth_bench_gold}
    question1, left = metrics.expand_scalar_gold(by_question["question1"], params)
    tvdi = [step for step in left if step["tool"] == "compute_tvdi"]
    assert len(tvdi) == 1
    assert len(set(tvdi[0]["lengths"].values())) > 1
    assert sum(call["name"] == "compute_tvdi" for call in question1) == 1

    published = next(call for call in by_question["question70"] if call["name"] == "compute_tvdi")
    expanded, unexpanded = metrics.expand_scalar_gold([published], params)
    assert unexpanded == []
    assert len(expanded) == 2
    assert expanded[0]["input"]["ndvi_path"] == published["input"]["ndvi_path"][0]
    assert expanded[1]["input"]["lst_path"] == published["input"]["lst_path"][1]
    assert isinstance(published["input"]["ndvi_path"], list)


def test_metric_edge_cases():
    call = lambda name, **kw: {"name": name, "input": kw}  # noqa: E731
    gold = [call("a", x=1), call("b"), call("c")]
    assert metrics.tool_any_order([call("c"), call("a")], gold) == pytest.approx(2 / 3)
    assert metrics.tool_in_order([call("a"), call("z"), call("c"), call("b")], gold) == pytest.approx(2 / 3)
    assert metrics.tool_exact_match([call("a", x=2), call("b"), call("z")], gold) == pytest.approx(2 / 3)
    assert metrics.parameter_accuracy([call("a", x=2), call("b"), call("c")], gold) == 0
    assert metrics.parameter_accuracy([call("a", x=1), call("b"), call("q")], gold) == pytest.approx(2 / 3)
    assert metrics.efficiency([call("a")] * 6, gold) == 2.0
    assert metrics.tool_exact_match([], [], upstream_compat=True) == 0.0
    assert metrics.tool_exact_match([], []) == 1.0
    assert metrics.efficiency([call("a")], []) == math.inf
    assert metrics.first_divergence_step([call("a"), call("z")], gold) == 2
    assert metrics.first_divergence_step(gold, gold) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<Answer>C<Answer>", "C"),
        ("blah <answer>b</answer>", "B"),
        ("I think A, final: D", "D"),
        ("FAIL to compute", "FAIL"),
        ("", "FAIL"),
        ("0.408", "UNKNOWN"),
    ],
)
def test_extract_answer(text, expected):
    assert metrics.extract_answer(text) == expected
