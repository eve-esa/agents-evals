from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agentic_evals.tasks.eo_gym import dataset

pytestmark = pytest.mark.upstream

PROMPTS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "eo_gym"
    / "upstream"
    / "eo-gym-evaluation"
    / "src"
    / "eo_gym_evaluation"
    / "prompts.py"
)


@pytest.fixture(scope="module")
def records():
    return dataset.load_records()


def test_prompts_are_verbatim():
    source = PROMPTS.read_text()
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SIMPLE_PROMPT", "COMPLEX_PROMPT"):
                call = node.value
                found[name] = call.func.value.value.strip()  # """...""".strip()
    assert found["SIMPLE_PROMPT"] == dataset.SIMPLE_PROMPT
    assert found["COMPLEX_PROMPT"] == dataset.COMPLEX_PROMPT


def test_samples(records):
    assert len(records) == 1436
    by_id = {r["id"]: r for r in records}
    sample = dataset.build_sample(by_id["feature_id_10003"], "simple", "skill")
    system, user = sample.input
    assert system.content == dataset.SIMPLE_PROMPT
    texts = [c.text for c in user.content if c.type == "text"]
    assert texts[0].startswith("Question: Question: [Before image: EO_GYM_DATA/")
    assert "Available image references:\n- Image 1: EO_GYM_DATA/" in texts[0]
    assert texts[1:] == ["Pre-disaster image:", "Post-disaster image:"]
    assert sum(c.type == "image" for c in user.content) == 2
    assert sample.target == "(25%, 50%]"
    assert sample.metadata["gold_steps"][0]["name"] == "get_building_mask_by_optical_image"
    assert "get_building_mask_by_optical_image" in sample.metadata["tools"]
    assert len(dataset.tools_for("xbd", "all")) == 35

    no_image = next(r for r in records if not r["images"])
    assert dataset.build_sample(no_image, "complex", "all").metadata["images"] == []
