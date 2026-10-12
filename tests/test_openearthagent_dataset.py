"""OpenEarthAgent samples from the committed test split."""

from __future__ import annotations

import pytest

from agentic_evals.tasks.openearthagent.dataset import load_samples
from agentic_evals.tasks.openearthagent.task import openearthagent


def test_e2e_split_matches_the_dataset_card():
    samples = load_samples("e2e")
    assert len(samples) == 1169
    assert len({sample.id for sample in samples}) == 1169
    image = [sample for sample in samples if sample.metadata["needs_vision"]]
    assert len(image) == 533
    assert sum(sample.metadata["is_generation"] for sample in samples) == 145
    assert all(sample.metadata["gold_names"][-1] == "Terminate" for sample in samples)


def test_a_known_gis_task():
    banff = load_samples("e2e", task_id="0")
    assert len(banff) == 1
    sample = banff[0]
    assert sample.metadata["question"].startswith("Which fire station")
    assert sample.metadata["gold_names"][0] == "GetAreaBoundary"
    assert "Banff Fire Station" in sample.metadata["reference_answer"]
    assert sample.metadata["reference_trajectory"][-1]["name"] != "Terminate"
    assert sample.metadata["needs_vision"] is False


def test_step_expansion_is_one_sample_per_gold_turn():
    samples = load_samples("step")
    # Dataset card: 7,064 reasoning steps in the test split. One is the
    # thought-only planning turn on every task, and those are exempt.
    assert len(samples) == 7064
    banff = [sample for sample in samples if sample.metadata["idx"] == "0"]
    assert banff[0].id == "0:1"
    assert banff[0].metadata["gold_text"].endswith('"actions": []}') or '"actions": []' in banff[0].metadata["gold_text"]
    assert banff[1].metadata["reference_trajectory"][0]["name"] == "GetAreaBoundary"
    assert banff[-1].target.startswith("Closest pair")
    assert banff[0].input[0].role == "system"
    assert "remote sensing assistant" in banff[0].input[0].text
    assert "<AGENT_PROMPT>" not in banff[0].input[1].text


def test_vision_task_refuses_a_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples("e2e", task_id="1", attach_images=True, image_root=tmp_path)


def test_step_task_builds_without_the_tool_server():
    built = openearthagent(regime="step", vision=False, task_id="0")
    assert built.metadata["regime"] == "step"
    assert built.metadata["openearthagent_rev"]
    assert len(built.dataset) == 6


def test_category_filters_keep_tasks_that_use_that_tool_family():
    from agentic_evals.tasks.openearthagent import metrics

    gis = load_samples("e2e", category="gis")
    perception = load_samples("e2e", category="perception")
    assert gis and perception
    assert len(gis) < 1169 and len(perception) < 1169
    assert any(sample.id == "0" for sample in gis)
    for sample in perception:
        names = [name for name in sample.metadata["gold_names"] if name != "Terminate"]
        assert any(metrics.tool_category(name) == "perception" for name in names)


def test_e2e_task_builds_without_images_when_vision_is_off():
    built = openearthagent(regime="e2e", vision=False, task_id="0")
    assert built.metadata["regime"] == "e2e"
    assert built.metadata["controller_url"]
    assert built.metadata["vision"] is False
    assert len(built.dataset) == 1
    assert built.headline_metric.scorer == "openearthagent_scorer"
    assert built.headline_metric.score == "any_order"
