"""Our Earth-Bench prompts must equal what upstream actually sent to the models."""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from agentic_evals.tasks.earth_bench import dataset
from conftest import published_run_dirs

pytestmark = pytest.mark.upstream


def _published_queries(root) -> dict[str, dict[int, str]]:
    """regime -> question number -> query, from every published run that got a real prompt."""
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for run_dir in published_run_dirs(root):
        regime = run_dir.name.rsplit("_", 1)[1].lower()
        for pred in json.loads((run_dir / "extracted_tool_calls.json").read_text()):
            query = pred["query"]
            if query.startswith("Error processing"):
                continue
            number = int(pred["question_index"].removeprefix("question"))
            previous = out[regime].setdefault(number, query)
            assert previous == query, (run_dir.name, number)
    return out


# Tasks whose published runs used benchmark/supported_data/, a directory that no
# longer exists upstream; the pinned question.json points them at benchmark/data/.
PATH_ONLY_CHANGES = {21, 22, 66, 67}
# Tasks that also had their question text revised in question.json after the
# published runs (all Products). Published scores for these are not comparable
# with runs on the pinned benchmark.
TEXT_CHANGES = {110, 121, 122, 124, 125, 141, 155, 156, 157, 182, 183, 184}


def test_prompts_match_published_queries(earth_agent_root):
    published = _published_queries(earth_agent_root)
    for regime in ("ap", "if"):
        samples = {s.id: s for s in dataset.load_samples(regime, "all", root=earth_agent_root)}
        assert len(samples) == 248
        assert set(published[regime]) == set(samples)
        path_only, text = set(), set()
        for number, query in published[regime].items():
            if samples[number].input == query:
                continue
            if samples[number].input == query.replace("benchmark/supported_data/", "benchmark/data/"):
                path_only.add(number)
            else:
                text.add(number)
        assert path_only == PATH_ONLY_CHANGES, regime
        assert text == TEXT_CHANGES, regime


def test_modalities_and_gold(earth_agent_root):
    counts = {m: len(dataset.load_samples("ap", m, root=earth_agent_root)) for m in ("spectrum", "products", "rgb")}
    assert counts == {"spectrum": 100, "products": 88, "rgb": 60}
    samples = {s.id: s for s in dataset.load_samples("if", "all", root=earth_agent_root)}
    assert samples[43].metadata["has_gold_trajectory"] is False
    assert samples[1].metadata["gold_tool_calls"][0] == {
        "name": "get_filelist",
        "input": {"dir_path": "benchmark/data/question1"},
    }
    assert samples[163].metadata["choices"] is None and samples[163].metadata["answer_question_json"] == "0.408"


# Tasks where question.json's whitelist and extracted_tool_calls_GT.json's
# final_answer disagree (both present). Pinned so an upstream change is noticed.
ANSWER_KEY_DISAGREEMENTS = {
    9, 107, 110, 132, 143, 204, 206, 207, 211, 212, 213, 214, 216, 217, 221, 226,
    227, 230, 232, 234, 237, 238, 239, 240, 241, 246, 247, 248,
}


def test_answer_keys(earth_agent_root):
    gt = {s.id: s for s in dataset.load_samples("ap", "all", root=earth_agent_root)}
    qj = {s.id: s for s in dataset.load_samples("ap", "all", answer_key="question_json", root=earth_agent_root)}
    disagreements = {
        n
        for n, s in gt.items()
        if s.metadata["answer_gt_file"] is not None and s.metadata["answer_gt_file"] != s.metadata["answer_question_json"]
    }
    assert disagreements == ANSWER_KEY_DISAGREEMENTS
    assert {n for n, s in gt.items() if s.metadata["answer_key_fallback"]} == {43, 164}
    assert all(s.target == s.metadata["answer_question_json"] for s in qj.values())
    assert gt[9].target == "A" and qj[9].target == "B"
