"""OpenEarthAgent metric functions against the released code's rules.

``summarize_step`` and ``summarize_e2e`` return the units ``evaluate_function``
returns (F1 and answer accuracy as percentages, tool-order as a fraction).
The differential check against that function runs when the cloned repo is
present (``tests/test_openearthagent_upstream.py``). There are no published
per-model trajectories, so this is weaker than Earth-Bench's oracle.
"""

from __future__ import annotations

import math

from agentic_evals.tasks.openearthagent import metrics


def _judge(gold, pred, question):
    del question
    return 1.0 if gold == pred else 0.0


def test_extract_action_reads_the_first_array_and_rejects_broken_json():
    text = '{"thought": "go", "actions": [{"name": "Calculator", "arguments": {"expression": "1+1"}}]}'
    assert metrics.extract_action(text) == [{"name": "Calculator", "arguments": {"expression": "1+1"}}]
    assert metrics.extract_action('{"thought": "plan", "actions": []}') == []
    assert metrics.extract_action("no actions here") is None
    assert metrics.extract_action('{"actions": [}') is None


def test_extract_tool_cfg_rejects_a_non_object_argument():
    cfg, error = metrics.extract_tool_cfg({"name": "OCR", "arguments": {"image": "img_1"}})
    assert error == ""
    assert cfg == {"API_name": "OCR", "API_params": {"image": "img_1"}}
    assert metrics.extract_tool_cfg({"name": "OCR", "arguments": "img_1"})[1] == "arguments_is_not_valid_dict"
    assert metrics.extract_tool_cfg(["OCR"])[1] == "action_is_not_valid_dict"


def test_extract_final_answer_and_score():
    text = '{"actions": [{"name": "Terminate", "arguments": {"ans": "  12 m  "}}]}'
    assert metrics.extract_final_answer(text) == "12 m"
    assert metrics.extract_final_answer("plain text") == ""
    assert metrics.extract_score('{"Score": 0.95, "Justification": "close, score 0.1"}') == 0.95
    assert metrics.extract_score("no number before Justification 0.2") == 0.0
    assert metrics.extract_score("") == 0.0


def test_compare_tool_sequences_keeps_multiset_and_unique_apart():
    # A repeated gold tool is not covered by one prediction. Unique still matches.
    same, any_order, unique = metrics.compare_tool_sequences(["Calculator", "Calculator"], ["Calculator"])
    assert (same, any_order, unique) == (False, False, True)
    # Multiset can match while the gold order is not a subsequence.
    same, any_order, unique = metrics.compare_tool_sequences(
        ["GetAreaBoundary", "AddPoisLayer"],
        ["AddPoisLayer", "GetAreaBoundary"],
    )
    assert any_order is True
    assert unique is True
    assert same is False
    same, any_order, unique = metrics.compare_tool_sequences(
        ["GetAreaBoundary", "AddPoisLayer"],
        ["GetAreaBoundary", "Calculator", "AddPoisLayer"],
    )
    assert (same, any_order, unique) == (True, True, True)


def test_category_f1_uses_the_released_smoothing():
    scores = metrics.category_f1(
        [["GetAreaBoundary", "AddPoisLayer", "Terminate"]],
        [["GetAreaBoundary", "AddPoisLayer"]],
    )
    correct = 2
    precision = correct / (2 + 1e-5)
    recall = correct / (2 + 1e-5)
    assert scores["gis"] == 2 * precision * recall / (precision + recall + 1e-5)
    assert scores["perception"] == 0.0
    # No calls at all stays defined, because of the 1e-5 terms.
    empty = metrics.category_f1([[]], [[]])
    assert empty["logic"] == 0.0
    assert not math.isnan(empty["logic"])


def test_step_sample_skips_the_planning_turn_and_scores_argument_names():
    planning = metrics.step_sample_score(
        '{"thought": "plan", "actions": []}',
        '{"actions": [{"name": "Calculator", "arguments": {"expression": "1"}}]}',
        is_generation_task=False,
    )
    assert planning["exempt"] is True
    gold = '{"actions": [{"name": "Calculator", "arguments": {"expression": "1+1"}}]}'
    wrong_value = metrics.step_sample_score(
        gold,
        '{"actions": [{"name": "Calculator", "arguments": {"expression": "2"}}]}',
        is_generation_task=False,
    )
    assert (wrong_value["inst"], wrong_value["tool"], wrong_value["argn"], wrong_value["argv"]) == (1, 1, 1, 0)
    missing_key = metrics.step_sample_score(
        gold,
        '{"actions": [{"name": "Calculator", "arguments": {}}]}',
        is_generation_task=False,
    )
    assert missing_key["argn"] == 0
    assert "required_argument_missing" in missing_key["errors"]


def test_step_answer_is_only_a_matching_terminate_on_a_non_generation_task():
    gold = '{"actions": [{"name": "Terminate", "arguments": {"ans": "836 m"}}]}'
    pred = '{"actions": [{"name": "Terminate", "arguments": {"ans": "836 m"}}]}'
    counted = metrics.step_sample_score(gold, pred, is_generation_task=False)
    assert counted["answer"] == ("836 m", "836 m")
    skipped = metrics.step_sample_score(gold, pred, is_generation_task=True)
    assert skipped["answer"] is None
    assert skipped["tool"] == 1
    wrong_tool = metrics.step_sample_score(
        gold,
        '{"actions": [{"name": "Calculator", "arguments": {"expression": "1"}}]}',
        is_generation_task=False,
    )
    assert wrong_tool["answer"] is None
    assert wrong_tool["tool"] == 0


def test_summarize_step_matches_the_released_percentages():
    rows = [
        {
            "gold_text": '{"actions": []}',
            "pred_text": '{"actions": []}',
            "is_generation": False,
            "question": "q",
        },
        {
            "gold_text": '{"actions": [{"name": "Calculator", "arguments": {"expression": "1"}}]}',
            "pred_text": '{"actions": [{"name": "Calculator", "arguments": {"expression": "1"}}]}',
            "is_generation": False,
            "question": "q",
        },
        {
            "gold_text": '{"actions": [{"name": "Terminate", "arguments": {"ans": "yes"}}]}',
            "pred_text": '{"actions": [{"name": "Terminate", "arguments": {"ans": "yes"}}]}',
            "is_generation": False,
            "question": "q",
        },
    ]
    summary = metrics.summarize_step(rows, _judge)
    # The planning turn is exempt. Two scored calls, both correct. One judged answer.
    assert summary["inst_acc"] == 100.0
    assert summary["tool_acc"] == 100.0
    assert summary["arg_acc2"] == 100.0
    assert summary["answer_acc"] == 100.0
    assert summary["count_non_gen_samples"] == 1


def test_summarize_e2e_splits_generation_tasks_and_drops_terminate_from_order():
    rows = [
        {
            "gold_names": ["GetAreaBoundary", "Terminate"],
            "pred_names": ["GetAreaBoundary"],
            "tool_response": [{"error_code": 0}],
            "prediction": "836 m",
            "gold_answer": "836 m",
            "question": "q",
            "n_rounds": 1,
        },
        {
            "gold_names": ["DrawBox", "Terminate"],
            "pred_names": ["DrawBox"],
            "tool_response": [{"error_code": 0}],
            "prediction": None,
            "gold_answer": "",
            "question": "q",
            "n_rounds": 2,
        },
    ]
    summary = metrics.summarize_e2e(rows, _judge)
    assert summary["answer_acc"] == 100.0
    assert summary["answer_acc_w_gen"] == 100.0
    assert summary["total_gen_samples"] == 1.0
    assert summary["all_tools_called_same_order"] == 1.0
    assert summary["avg_steps_taken"] == 1.5
    assert summary["tool_call_error"] == 0.0


def test_summarize_e2e_does_not_index_a_short_gold_trajectory():
    rows = [
        {
            "gold_names": ["Terminate"],
            "pred_names": [],
            "tool_response": [],
            "prediction": "",
            "gold_answer": "x",
            "question": "q",
            "n_rounds": 0,
        }
    ]
    summary = metrics.summarize_e2e(rows, _judge)
    assert summary["answer_acc"] == 0.0
    assert summary["total_gen_samples"] == 0.0


def test_empty_prediction_does_not_call_the_judge():
    called = {"n": 0}

    def judge(gold, pred, question):
        del gold, pred, question
        called["n"] += 1
        return 1.0

    assert metrics.judge_or_zero("x", "", "q", judge) == 0.0
    assert metrics.judge_or_zero("x", 0, "q", judge) == 0.0
    assert called["n"] == 0


def test_is_generation_needs_the_tool_before_terminate():
    assert metrics.is_generation(["DrawBox", "Terminate"]) is True
    assert metrics.is_generation(["Terminate"]) is False
    assert metrics.is_generation([]) is False
