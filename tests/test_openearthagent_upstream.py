"""Differential check against ``evaluate_function`` in the cloned tool server.

OpenEarthAgent publishes no per-model trajectories. This runs the released
aggregator on synthetic rows and compares it to ``summarize_step`` /
``summarize_e2e``. The judge is a stub on both sides. The test skips until
``scripts/setup_benchmark.py openearthagent`` has cloned the pinned revision
(the GitHub repo has no license, so that file is not committed here).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agentic_evals.paths import OPENEARTHAGENT_UPSTREAM
from agentic_evals.tasks.openearthagent import metrics

pytestmark = pytest.mark.upstream

_TASK = Path("tool_server/tf_eval/tasks/rsagent/task.py")
_WANTED = {
    "extract_action",
    "extract_tool_cfg",
    "extract_final_answer",
    "extract_score",
    "gettype",
    "get_all_cfg_from_conversation",
    "last_tool_in_gen",
    "compare_tool_sequences",
    "evaluate_function",
}


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


def _load_upstream(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOLS_LIST":
                    parts.append(ast.get_source_segment(source, node) or "")
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            parts.append(ast.get_source_segment(source, node) or "")
    namespace: dict = {"Counter": __import__("collections").Counter, "re": __import__("re"), "json": json, "os": __import__("os"), "logger": _Logger()}
    exec("\n\n".join(parts), namespace)  # noqa: S102 - the released functions, isolated from their imports
    namespace["sim_score"] = lambda gold, pred, question: 1.0 if gold == pred else 0.0
    return namespace


@pytest.fixture(scope="module")
def upstream():
    path = OPENEARTHAGENT_UPSTREAM / _TASK
    if not path.is_file():
        pytest.skip("OpenEarthAgent clone missing; run `uv run python scripts/setup_benchmark.py openearthagent --skip-data`")
    return _load_upstream(path)


def _judge(gold, pred, question):
    del question
    return 1.0 if gold == pred else 0.0


def test_step_aggregate_matches_evaluate_function(upstream):
    conversation = [
        {"from": "human", "value": "<AGENT_PROMPT>\n\nQuestion: How many?"},
        {"from": "gpt", "value": '{"thought": "plan", "actions": []}'},
        {"from": "human", "value": "How many?"},
        {
            "from": "gpt",
            "value": '{"actions": [{"name": "Calculator", "arguments": {"expression": "1+1"}}]}',
        },
        {"from": "human", "value": "OBSERVATION: 2"},
        {
            "from": "gpt",
            "value": '{"actions": [{"name": "Terminate", "arguments": {"ans": "2"}}]}',
        },
    ]
    turns = [
        ('{"thought": "plan", "actions": []}', '{"actions": []}'),
        (
            '{"actions": [{"name": "Calculator", "arguments": {"expression": "1+1"}}]}',
            '{"actions": [{"name": "Calculator", "arguments": {"expression": "1+1"}}]}',
        ),
        (
            '{"actions": [{"name": "Terminate", "arguments": {"ans": "2"}}]}',
            '{"actions": [{"name": "Terminate", "arguments": {"ans": "2"}}]}',
        ),
    ]
    meta = []
    results = []
    rows = []
    for index, (gold, pred) in enumerate(turns):
        idx = f"s{index}"
        meta.append({"idx": idx, "label": gold, "conversation": conversation})
        results.append({"idx": idx, "results": {"model_output": [pred]}})
        rows.append({"gold_text": gold, "pred_text": pred, "is_generation": False, "question": "How many?"})
    upstream["task_config"] = {"task_type": "step"}
    got = upstream["evaluate_function"](results, meta)["metrics"]
    ours = metrics.summarize_step(rows, _judge)
    for key in ("inst_acc", "tool_acc", "arg_acc1", "arg_acc2", "answer_acc", "count_non_gen_samples", "count_gen_samples"):
        assert ours[key] == pytest.approx(got[key])


def test_e2e_aggregate_matches_evaluate_function(upstream):
    conversation = [
        {"from": "human", "value": "<AGENT_PROMPT>\n\nQuestion: Where?"},
        {"from": "gpt", "value": '{"actions": [{"name": "GetAreaBoundary", "arguments": {"area": "Banff"}}]}'},
        {"from": "gpt", "value": '{"actions": [{"name": "Terminate", "arguments": {"ans": "Banff"}}]}'},
    ]
    meta = [
        {"idx": "e0", "label": "Banff", "conversation": conversation},
        {"idx": "e1", "label": "map", "conversation": conversation},
    ]
    results = [
        {
            "idx": "e0",
            "results": {
                "final_answer": "Banff",
                "current_round": 1,
                "model_response": ["call", "done"],
                "action_parse_error": ["", ""],
                "tool_cfg": [[{"API_name": "GetAreaBoundary", "API_params": {"area": "Banff"}}]],
                "tool_response": [{"text": "saved", "error_code": 0}],
            },
        },
        {
            "idx": "e1",
            "results": {
                "final_answer": None,
                "current_round": 1,
                "model_response": ["call"],
                "action_parse_error": [""],
                "tool_cfg": [[{"API_name": "DrawBox", "API_params": {"image": "img_1"}}]],
                "tool_response": [{"text": "drew", "error_code": 0}],
            },
        },
    ]
    # Gold for e1 must be a generation task. The conversation above is shared and
    # is not a generation task, so give e1 its own conversation.
    meta[1]["conversation"] = [
        {"from": "human", "value": "<AGENT_PROMPT>\n\nQuestion: Draw it."},
        {"from": "gpt", "value": '{"actions": [{"name": "DrawBox", "arguments": {"image": "img_1"}}]}'},
        {"from": "gpt", "value": '{"actions": [{"name": "Terminate", "arguments": {"ans": "map"}}]}'},
    ]
    rows = [
        {
            "gold_names": ["GetAreaBoundary", "Terminate"],
            "pred_names": ["GetAreaBoundary"],
            "tool_response": [{"error_code": 0}],
            "prediction": "Banff",
            "gold_answer": "Banff",
            "question": "Where?",
            "n_rounds": 1,
        },
        {
            "gold_names": ["DrawBox", "Terminate"],
            "pred_names": ["DrawBox"],
            "tool_response": [{"error_code": 0}],
            "prediction": None,
            "gold_answer": "map",
            "question": "Draw it.",
            "n_rounds": 1,
        },
    ]
    upstream["task_config"] = {"task_type": "e2e"}
    got = upstream["evaluate_function"](results, meta)["metrics"]
    ours = metrics.summarize_e2e(rows, _judge)
    for key in (
        "answer_acc",
        "answer_acc_w_gen",
        "perception_f1",
        "operation_f1",
        "logic_f1",
        "gis_f1",
        "all_tools_called_any_order",
        "all_tools_called_same_order",
        "all_tools_called_unique",
        "avg_steps_taken",
        "total_gen_samples",
    ):
        assert ours[key] == pytest.approx(got[key]), key
