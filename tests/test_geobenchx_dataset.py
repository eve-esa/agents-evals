"""GeoBenchX samples and solution rendering against the pinned checkout."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from agentic_evals.tasks.geobenchx import dataset, metrics
from agentic_evals.tasks.geobenchx.dataset import ASSETS

pytestmark = pytest.mark.upstream

TASKS = ASSETS / "tasks_and_reference_solutions.json"
TUNING = ASSETS / "evaluator_tuning_set.json"
UPSTREAM = Path(__file__).resolve().parent / "fixtures" / "geobenchx" / "upstream"


@pytest.fixture(scope="module")
def samples():
    return dataset.load_samples(on=date(2020, 1, 2))


def test_prompts_match_upstream():
    source = (UPSTREAM / "geobenchx" / "prompts.py").read_text(encoding="utf-8")
    source = source.replace("current_date = date.today()", "current_date = date(2020, 1, 2)")
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - their prompt module, date pinned above
    assert namespace["RULES_PROMPT"] == dataset.RULES_PROMPT
    assert namespace["SYSTEM_PROMPT"] + namespace["RULES_PROMPT"] == dataset.system_prompt(date(2020, 1, 2))


def test_sample_counts_and_split(samples):
    assert len(samples) == 202
    kinds = Counter(sample.metadata["kind"] for sample in samples)
    assert kinds == {"solvable": 120, "unsolvable": 79, "control": 3}
    assert sum(len(sample.metadata["reference_solutions"]) > 1 for sample in samples) == 17
    assert len(dataset.load_samples(group="merge")) == 36
    assert len(dataset.load_samples(group="process")) == 56
    assert len(dataset.load_samples(group="spatial")) == 53
    assert len(dataset.load_samples(group="heatmaps")) == 54
    assert len(dataset.load_samples(kind="unsolvable", group="spatial")) == sum(
        1 for sample in samples if sample.metadata["kind"] == "unsolvable" and "Spatial operations" in sample.metadata["labels"]
    )


def test_sample_shape(samples):
    sample = next(s for s in samples if s.metadata["kind"] == "control")
    system, user = sample.input
    assert system.role == "system"
    assert "Today is 2020-01-02." in system.content
    assert dataset.RULES_PROMPT in system.content
    assert user.content == user.text
    assert sample.metadata["reference_solutions"] == [[]]
    assert sample.metadata["reference_trajectory"] == []
    assert sample.metadata["prompt_date"] == "2020-01-02"

    multi = next(s for s in samples if len(s.metadata["reference_solutions"]) > 1)
    first = multi.metadata["reference_solutions"][0]
    assert multi.metadata["reference_trajectory"] == [
        {"name": step["name"], "arguments": step["arguments"], "output": None} for step in first
    ]
    one = dataset.load_samples(task_id=sample.id)
    assert [s.id for s in one] == [sample.id]


def test_tuning_set_has_human_and_llm_scores():
    if not TUNING.exists():
        pytest.skip("GeoBenchX checkout missing")
    tasks = json.loads(TUNING.read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) == 50
    assert {task["match_score_Human"] for task in tasks} <= {0, 1, 2}
    assert {task["match_score_LLM"] for task in tasks} <= {0, 1, 2}
    assert all(task.get("generated_solution") for task in tasks)


def test_task_records_the_pinned_revision():
    if not TASKS.exists():
        pytest.skip("GeoBenchX checkout missing")
    from agentic_evals.tasks.geobenchx.task import geobenchx

    built = geobenchx(group="heatmaps", kind="unsolvable")
    assert str(built.metadata["geobenchx_rev"]).startswith("0edf6109")
    assert built.metadata["recursion_limit"] == 25
    assert built.config.temperature == 0
    assert all(sample.metadata["kind"] == "unsolvable" for sample in built.dataset)
    assert all("Heatmaps, Contour Lines" in sample.metadata["labels"] for sample in built.dataset)


def test_exported_judge_prompt_keeps_its_placeholders():
    text = (ASSETS / "judge_prompt.txt").read_text(encoding="utf-8")
    for field in ("{task_text}", "{reference_solutions}", "{candidate_solution}", "<MATCH SCORE>", "<MATCH REASONING>"):
        assert field in text


def test_mcp_server_exposes_the_agent_tools():
    pytest.importorskip("geopandas")
    from agentic_evals.tasks.geobenchx.tools import mcp_server

    names = [tool.name for tool in mcp_server.TOOLS]
    schema = mcp_server.input_schema(next(tool for tool in mcp_server.TOOLS if tool.name == "load_data"))
    reject = next(tool for tool in mcp_server.TOOLS if tool.name == "reject_task")
    payload = {
        "names": names,
        "state_in_schema": "state" in schema.get("properties", {}),
        "reject": reject.func(state=mcp_server.STATE),
    }
    assert len(payload["names"]) == 24
    assert "plot_contour_lines" not in payload["names"]
    assert payload["names"][-2:] == ["reject_task", "calculate_column_statistics"]
    assert payload["state_in_schema"] is False
    assert payload["reject"] == "The task is not solvable with the tools and datasets available."


def test_solution_code_matches_upstream_get_solution_code():
    pytest.importorskip("pandas")
    sys.path.insert(0, str(UPSTREAM))
    from geobenchx.dataclasses import Solution, Step
    from geobenchx.utils import get_solution_code

    def render(steps, add_num=False):
        solution = Solution(steps=[
            Step(function_name=step["function_name"], arguments=step["arguments"], comment=step.get("comment"))
            for step in steps
        ])
        return get_solution_code(solution, add_num=add_num)

    tuning = json.loads(TUNING.read_text(encoding="utf-8"))
    main = json.loads(TASKS.read_text(encoding="utf-8"))
    rows = []
    for task in tuning["tasks"]:
        steps = task["generated_solution"]["steps"]
        rows.append({"steps": steps, "add_num": False, "code": render(steps)})
    commented = next(
        solution["steps"]
        for task in main["tasks"]
        for solution in task["reference_solutions"]
        if any(step.get("comment") for step in solution["steps"])
    )
    rows.append({"steps": commented, "add_num": False, "code": render(commented)})
    rows.append({"steps": commented, "add_num": True, "code": render(commented, add_num=True)})
    for row in rows:
        ours = metrics.solution_code(
            [
                {"name": step["function_name"], "arguments": step["arguments"], "comment": step.get("comment")}
                for step in row["steps"]
            ],
            add_num=row["add_num"],
        )
        assert ours == row["code"]
