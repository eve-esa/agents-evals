"""GeoBenchX metric functions against the released code's rules.

The judge itself is not called. ``solution_code`` is checked against
``get_solution_code`` in GeoBenchX's venv when that venv exists
(``tests/test_geobenchx_dataset.py``).
"""

from __future__ import annotations

import fnmatch
import math
import re

from inspect_ai.scorer import SampleScore, Score

from agentic_evals.tasks.geobenchx import metrics
from agentic_evals.tasks.geobenchx.scorers import (
    by_kind,
    display_model_id,
    is_expired_request,
    match_values,
    median,
    merged_match,
    retry_delay,
)


def step(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments, "comment": None}


def test_solution_code_quotes_every_value_and_skips_empty_comments():
    text = metrics.solution_code(
        [
            {
                "name": "filter_categorical",
                "arguments": {"dataframe_name": "countries", "filters": {"SUBREGION": "Eastern Africa"}},
                "comment": "Also acceptable filter: 'Country Name'",
            },
            {"function_name": "reject_task", "arguments": {}, "comment": None},
        ]
    )
    assert text == (
        "filter_categorical(dataframe_name='countries', filters='{'SUBREGION': 'Eastern Africa'}') "
        "#Also acceptable filter: 'Country Name' \n"
        "reject_task() \n"
    )
    numbered = metrics.solution_code([step("load_data", dataset="gdp")], add_num=True)
    assert numbered == "1. load_data(dataset='gdp') \n"


def test_reference_block_includes_a_description_only_when_set():
    solutions = [[step("reject_task")]]
    assert metrics.reference_solutions_text(None, solutions) == " \nreject_task() \n\n "
    described = metrics.reference_solutions_text("", solutions)
    assert described.startswith("Here are general comments about the reference solutions: ")


def test_parse_verdict():
    text = "<MATCH REASONING>\npartial, colormap differs\n</MATCH REASONING>\n<MATCH SCORE>1</MATCH SCORE>"
    assert metrics.parse_verdict(text) == (1, "partial, colormap differs")
    assert metrics.parse_verdict("<MATCH SCORE>3</MATCH SCORE>")[0] is None
    assert metrics.parse_verdict("no tags")[0] is None
    assert metrics.parse_verdict("<MATCH SCORE>nope</MATCH SCORE>")[0] is None


def test_length_diff_follows_generate_eval_stats():
    # (4 - mean(2, 4)) / 3 = 2/3. Two references, mean length 3.
    assert metrics.length_diff(4, [[step("a"), step("b")], [step("a")] * 4]) == (4 - 3) / 3
    # Mean reference length 0: the difference is the candidate length, not a ratio.
    assert metrics.length_diff(2, [[]]) == 2
    assert metrics.length_diff(0, [[]]) == 0
    assert metrics.length_diff(5, []) == 5


def test_step_scores_take_the_max_over_references():
    pred = [step("load_data", dataset="gdp"), step("load_geodata", geodataset="countries")]
    close = [step("load_data", dataset="gdp"), step("load_geodata", geodataset="countries")]
    far = [step("reject_task")]
    scores = metrics.step_scores(pred, [far, close])
    assert scores == {"tao": 1.0, "tio": 1.0, "tem": 1.0, "param_acc": 1.0}
    # A worse argument stops the parameter prefix and does not affect TAO.
    wrong_arg = [step("load_data", dataset="co2"), step("load_geodata", geodataset="countries")]
    partial = metrics.step_scores(pred, [wrong_arg])
    assert partial["tao"] == 1.0
    assert partial["tem"] == 1.0
    assert partial["param_acc"] == 0.0


def test_empty_reference_matches_only_an_empty_candidate():
    assert metrics.step_scores([], [[]]) == {"tao": 1.0, "tio": 1.0, "tem": 1.0, "param_acc": 1.0}
    assert metrics.step_scores([step("load_data")], [[]])["tao"] == 0.0


def test_by_kind_and_median():
    grouped = by_kind()(
        [
            SampleScore(score=Score(value=1.0), sample_metadata={"kind": "solvable"}),
            SampleScore(score=Score(value=0.0), sample_metadata={"kind": "solvable"}),
            SampleScore(score=Score(value=1.0), sample_metadata={"kind": "unsolvable"}),
        ]
    )
    assert grouped["solvable"] == 0.5
    assert grouped["unsolvable"] == 1.0
    assert math.isnan(grouped["control"])
    assert median()(
        [SampleScore(score=Score(value=value)) for value in (1.0, 3.0, 2.0, 4.0)]
    ) == {"median": 2.5}


class _Err(Exception):
    def __init__(self, message: str, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def test_retry_delay_doubles_and_caps_at_one_minute():
    assert [retry_delay(attempt) for attempt in range(1, 7)] == [2, 4, 8, 16, 32, 60]


def test_only_an_expired_or_timed_out_400_is_retried_outside_inspect():
    assert is_expired_request(_Err("The request has been expired", status_code=400))
    assert is_expired_request(_Err("Request timed out.", status_code=400))
    assert is_expired_request(_Err("provider returned a completion (expired)", code="invalid_request"))
    assert not is_expired_request(_Err("maximum context length is 40960 tokens", status_code=400))
    assert not is_expired_request(_Err("temporarily rate-limited upstream", status_code=429))
    assert not is_expired_request(_Err("upstream connect error", status_code=502))


def test_match_values_are_named_with_the_judge():
    values = match_values("openai-api/openrouter/qwen/qwen3.8-27b", 2, 0.5)
    assert display_model_id("openai-api/openrouter/qwen/qwen3.8-27b") == "qwen/qwen3.8-27b"
    assert values == {"qwen/qwen3.8-27b/match": 2, "qwen/qwen3.8-27b/length_diff": 0.5}
    # Inspect expands metric globs with re.compile(fnmatch.translate(...)).
    match_key = re.compile(fnmatch.translate("*/match"))
    assert match_key.match("qwen/qwen3.8-27b/match")
    assert match_key.match("qwen/qwen3.8-27b/length_diff") is None
    missing = match_values("grader", None, 1.0)
    assert math.isnan(missing["grader/match"])
    assert missing["grader/length_diff"] == 1.0


def test_merged_match_skips_nan_and_does_not_average_length():
    qwen = match_values("qwen/qwen3.8-27b", 2, 0.5)
    llama = match_values("meta-llama/llama-4-maverick", None, 0.5)
    merged = merged_match([qwen, llama])
    assert merged == {"merged/match": 2.0}
    assert "merged/length_diff" not in merged
    both = merged_match([match_values("a", 2, 0.0), match_values("b", 0, 1.0)])
    assert both["merged/match"] == 1.0
    assert "merged/length_diff" not in both
    empty = merged_match([])
    assert math.isnan(empty["merged/match"])
    assert "merged/length_diff" not in empty


def test_reject_task_after_other_calls_is_not_a_perfect_prefix():
    # Their judge would score this a full match. The step metrics do not.
    pred = [step("load_data", dataset="gdp"), step("reject_task")]
    scores = metrics.step_scores(pred, [[step("reject_task")]])
    assert scores["tao"] == 1.0
    assert scores["tem"] == 0.0
    assert scores["param_acc"] == 0.0
