"""Differential test: our EO-Gym metrics vs the released ``agent_passk.py``.

EO-Gym publishes no per-model outputs, so there is no oracle like Earth-Bench's.
Instead we build synthetic trajectories from real test-set records (dropped,
reordered and extra steps, wrong arguments, error observations, missing
``Final Answer:`` markers, varying correctness patterns), score them with both
implementations, and require identical per-question rows and summary rows. The
judge is replaced by a deterministic stub on both sides; judging itself is
tested separately.
"""

from __future__ import annotations

import json
import math
import random
import subprocess

import pytest

from agentic_evals.paths import EO_GYM_ROOT, EO_GYM_VENV, venv_python
from agentic_evals.tasks.eo_gym import metrics

pytestmark = pytest.mark.upstream

TEST_SET = EO_GYM_ROOT / "datasets" / "eo_gym_test_set.jsonl"

# Runs inside EO-Gym's venv: agent_passk imports EO-Gym's tool registry (torch etc.).
UPSTREAM_SCRIPT = r"""
import json, sys
sys.path[:0] = [sys.argv[1] + "/eo-gym-evaluation/src", sys.argv[1] + "/eo-gym-data-synthesis/src"]
from pathlib import Path
# Released defect: agent_passk imports these aliases from prompt_manager, which
# defines them in evaluation_prompts but does not re-export them.
import eo_gym_data_synthesis.prompt_manager as pm
from eo_gym_data_synthesis.prompt_manager import evaluation_prompts as ep
pm.SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT = ep.SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT
pm.TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT = ep.TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT
from eo_gym_evaluation import agent_passk

def norm(t):
    return " ".join(str(t or "").strip().split()).lower()

class StubMatcher:
    def judge(self, *, question, gt_answer, pred_answer):
        return norm(gt_answer) == norm(pred_answer), "stub", False, False

ev = agent_passk._evaluate_prediction_file(
    path=Path(sys.argv[3]), gt_map=agent_passk._load_ground_truth(Path(sys.argv[2])),
    uid_question_type_map={}, semantic_matcher=StubMatcher(), max_traj=3,
    max_questions=None, semantic_workers=1, show_progress=False,
)
summary = agent_passk._build_summary_rows(evaluations=[ev])[0]
print(json.dumps({"rows": ev.question_rows, "summary": summary}, default=str))
"""


class StubMatcher:
    """Correct iff the candidate answer is the reference answer (same rule as the upstream side)."""

    def judge(self, *, question, gt_answer, pred_answer):
        ok = metrics.normalize_text(gt_answer) == metrics.normalize_text(pred_answer)
        return ok, "stub", False, False


@pytest.fixture(scope="module")
def upstream_python():
    if not TEST_SET.exists():
        pytest.skip("EO-Gym checkout missing; run `uv run python scripts/setup_benchmark.py eo-gym --skip-data`")
    python = venv_python(EO_GYM_VENV)
    if not python.exists():
        pytest.skip("EO-Gym venv missing; run `uv run python scripts/setup_benchmark.py eo-gym --skip-data`")
    return python


def _perturb(gold: list[metrics.Step], rng: random.Random) -> list[metrics.Step]:
    steps = [metrics.Step(s.name, s.arguments, s.observation) for s in gold]
    ops = rng.sample(["drop", "swap", "extra", "args", "error", "none", "none"], k=2)
    for op in ops:
        if op == "drop" and steps:
            steps.pop(rng.randrange(len(steps)))
        elif op == "swap" and len(steps) > 1:
            i = rng.randrange(len(steps) - 1)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        elif op == "extra":
            steps.insert(rng.randrange(len(steps) + 1), metrics.Step("basic_calculator", '{"expression": "1+1"}', "2"))
        elif op == "args" and steps:
            i = rng.randrange(len(steps))
            steps[i] = metrics.Step(steps[i].name, '{"x": "different"}', steps[i].observation)
        elif op == "error" and steps:
            i = rng.randrange(len(steps))
            steps[i] = metrics.Step(steps[i].name, steps[i].arguments, "Error: tool failed")
    return steps


def _synthesize(records, rng):
    """For each record, 3 trajectories as (steps, final_message)."""
    out = {}
    for rec in records:
        gold = metrics.gold_steps_from_messages(rec["messages"])
        trajs = []
        pattern = rng.choice(["ccc", "icc", "iic", "iii", "cii", "ici"])
        for k, flag in enumerate(pattern):
            steps = gold if (k == 0 and rng.random() < 0.3) else _perturb(gold, rng)
            answer = rec["gt_answer"] if flag == "c" else "something else"
            final = f"Reasoning...\nFinal Answer: {answer}" if rng.random() < 0.8 else answer
            trajs.append((steps, final))
        out[rec["id"]] = trajs
    return out


def _to_upstream_record(rec, traj_id, steps, final, question):
    conversations = [{"from": "human", "value": question}]
    for s in steps:
        conversations.append({"from": "function_call", "value": json.dumps({"name": s.name, "arguments": s.arguments})})
        conversations.append({"from": "observation", "value": s.observation})
    conversations.append({"from": "gpt", "value": final})
    return {
        "uid": f"{rec['id']}:traj{traj_id}",
        "id": rec["id"],
        "tool_schema_mode": "dataset_mapped",
        "tool_call_mode": "builtin_function_call",
        "conversations": conversations,
        "question": question,
        "question_type": rec["question_type"],
        "data_source": rec["data_source"],
    }


def test_matches_released_scorer(upstream_python, tmp_path):
    rng = random.Random(0)
    records = [json.loads(line) for line in TEST_SET.read_text().splitlines()]
    records = rng.sample(records, 150)
    synthetic = _synthesize(records, rng)

    pred_path = tmp_path / "test_openai_model_simple_prompt.jsonl"
    with pred_path.open("w") as f:
        for rec in records:
            question = metrics.judge_question_from_messages(rec["messages"])
            for k, (steps, final) in enumerate(synthetic[rec["id"]]):
                f.write(json.dumps(_to_upstream_record(rec, k, steps, final, question)) + "\n")

    result = subprocess.run(
        [str(upstream_python), "-c", UPSTREAM_SCRIPT, str(EO_GYM_ROOT), str(TEST_SET), str(pred_path)],
        capture_output=True,
        text=True,
        env={"RUN_ON_LOCAL": "1", "TOOL_RUN_ON_LOCAL": "1", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr[-3000:]
    upstream = json.loads(result.stdout.strip().splitlines()[-1])
    theirs_rows = {row["question_id"]: row for row in upstream["rows"]}
    theirs_summary = upstream["summary"]

    our_rows = {}
    for rec in records:
        gold = metrics.gold_steps_from_messages(rec["messages"])
        trajs = []
        for k, (steps, final) in enumerate(synthetic[rec["id"]]):
            answer, missing = metrics.extract_final_answer(final)
            tm = metrics.tool_metrics(gold, steps)
            trajs.append(
                metrics.Trajectory(
                    traj_id=k,
                    correct=StubMatcher().judge(question="", gt_answer=rec["gt_answer"], pred_answer=answer)[0],
                    function_calls=len(steps),
                    function_call_errors=sum(s.is_error for s in steps),
                    missing_marker=missing,
                    tools_any_order=tm["tools_any_order"],
                    tools_in_order=tm["tools_in_order"],
                    tools_exact_match=tm["tools_exact_match"],
                    parameters_score=tm["parameters_score"],
                    reference_steps=len(gold),
                )
            )
        our_rows[rec["id"]] = metrics.early_stop(trajs, max_traj=3)

    assert set(our_rows) == set(theirs_rows)
    for qid, ours in our_rows.items():
        theirs = theirs_rows[qid]
        for key in ("accuracy", "pass@1", "pass@2", "pass@3", "tries_used", "reference_steps"):
            assert ours[key] == pytest.approx(float(theirs[key])), (qid, key)
        assert ours["consumed_tool_calls"] == theirs["consumed_tool_calls"], qid
        assert ours["consumed_function_call_errors"] == theirs["consumed_function_call_errors"], qid
        assert ours["consumed_missing_markers"] == theirs["consumed_missing_final_answer_markers"], qid

    rows = list(our_rows.values())
    ours_summary = metrics.pooled_summary(rows)
    for key, value in ours_summary.items():
        assert value == pytest.approx(theirs_summary[key]), key
    for key in ("accuracy", "pass@1", "pass@2", "pass@3"):
        assert sum(r[key] for r in rows) / len(rows) == pytest.approx(theirs_summary[key]), key


def test_unbiased_matches_inspect_pass_at():
    from inspect_ai.scorer import Score
    from inspect_ai.scorer._reducer.reducer import pass_at

    for n, c, k in [(3, 0, 1), (3, 1, 1), (3, 1, 2), (3, 2, 2), (3, 3, 3), (10, 4, 5)]:
        scores = [Score(value=1.0 if i < c else 0.0) for i in range(n)]
        assert metrics.unbiased_pass_at_k(n, c, k) == pytest.approx(pass_at(k)(scores).value), (n, c, k)


def test_early_stop_differs_from_unbiased_on_pass1():
    t = lambda i, ok: metrics.Trajectory(i, ok, 1, 0, False, 1, 1, 1, 1.0, 1)  # noqa: E731
    row = metrics.early_stop([t(0, False), t(1, True), t(2, True)])
    assert row["pass@1"] == 0.0 and row["unbiased_pass@1"] == pytest.approx(2 / 3)
    assert row["pass@3"] == 1.0 and row["unbiased_pass@3"] == 1.0
    assert row["consumed_trajectories"] == 2 and not math.isnan(row["unbiased_pass@2"])


def test_truncated_answer_after_tools():
    from agentic_evals.agents.langgraph_react import REACHED_MAXIMUM_ITERATIONS, truncated_final_answer

    limit = "Model call limits exceeded: run limit (15/15)"
    assert truncated_final_answer(limit, tool_calls=3) == REACHED_MAXIMUM_ITERATIONS
    assert truncated_final_answer(limit, tool_calls=0) is None
    assert truncated_final_answer("Final Answer: 5", tool_calls=3) is None


def test_judge_schema_requires_semantic_fields():
    from agentic_evals.tasks.eo_gym.scorers import judge_generate_config

    config = judge_generate_config()
    assert config.temperature == 0
    schema = config.response_schema
    assert schema is not None and schema.strict is False
    body = schema.json_schema
    assert body.type == "object"
    assert set(body.required or []) >= {"is_same_meaning", "reason"}
    assert body.properties is not None
    assert body.properties["is_same_meaning"].type == "boolean"
    assert body.properties["reason"].type == "string"
