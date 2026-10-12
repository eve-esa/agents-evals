"""Each task's default iteration budget, and a YAML-equivalent override."""

from __future__ import annotations

import pytest

from agentic_evals.tasks.earth_bench.task import earth_bench
from agentic_evals.tasks.eo_gym.task import eo_gym
from agentic_evals.tasks.geobenchx.task import geobenchx
from agentic_evals.tasks.openearthagent.task import openearthagent


def _closed(fn, name: str):
    """Value of a free variable on ``fn`` or a callable it closes over."""
    found = []

    def walk(current) -> None:
        code = getattr(current, "__code__", None)
        closure = getattr(current, "__closure__", None)
        if code is None or not closure:
            return
        for free, cell in zip(code.co_freevars, closure, strict=True):
            value = cell.cell_contents
            if free == name:
                found.append(value)
            if callable(value):
                walk(value)

    walk(fn)
    if not found:
        raise AssertionError(f"{name} is not closed over by {fn}")
    return found[-1]


def test_earth_bench_defaults_to_fifty_supersteps():
    built = earth_bench()
    assert built.metadata["max_iterations"] == 50
    assert _closed(built.solver, "recursion_limit") == 50

    overridden = earth_bench(max_iterations=7)
    assert overridden.metadata["max_iterations"] == 7
    assert _closed(overridden.solver, "recursion_limit") == 7


def test_geobenchx_defaults_to_twenty_five_supersteps():
    built = geobenchx(group="heatmaps", kind="unsolvable")
    assert built.metadata["max_iterations"] == 25
    assert "recursion_limit" not in built.metadata
    assert _closed(built.solver, "recursion_limit") == 25

    overridden = geobenchx(group="heatmaps", kind="unsolvable", max_iterations=7)
    assert overridden.metadata["max_iterations"] == 7
    assert _closed(overridden.solver, "recursion_limit") == 7


def test_eo_gym_defaults_to_fifteen_model_turns():
    built = eo_gym(vision=False)
    assert built.metadata["max_iterations"] == 15
    assert "max_calls" not in built.metadata

    overridden = eo_gym(vision=False, max_iterations=4)
    assert overridden.metadata["max_iterations"] == 4


def test_eo_gym_wires_the_model_turn_cap(monkeypatch):
    from inspect_ai.dataset import MemoryDataset, Sample

    monkeypatch.setattr(
        "agentic_evals.tasks.eo_gym.task.eo_gym_dataset",
        lambda *args, **kwargs: MemoryDataset([Sample(input="q", target="a")]),
    )
    built = eo_gym(vision=True, max_iterations=4)
    assert built.metadata["max_iterations"] == 4
    assert _closed(built.solver, "max_model_calls") == 4
    assert _closed(built.solver, "recursion_limit") == 4 * 4 + 10


def test_openearthagent_both_regimes_default_to_fifteen():
    e2e = openearthagent(regime="e2e", vision=False, task_id="0")
    assert e2e.metadata["max_iterations"] == 15
    assert "max_rounds" not in e2e.metadata
    assert e2e.solver.__registry_info__.name == "agentic_evals/langgraph_react"
    assert _closed(e2e.solver, "max_model_calls") == 15
    assert _closed(e2e.solver, "recursion_limit") == 4 * 15 + 10

    step = openearthagent(regime="step", vision=False, task_id="0")
    assert step.metadata["max_iterations"] == 15
    assert step.solver.__registry_info__.name == "inspect_ai/generate"

    overridden = openearthagent(regime="e2e", vision=False, task_id="0", max_iterations=9)
    assert overridden.metadata["max_iterations"] == 9
    assert _closed(overridden.solver, "max_model_calls") == 9

    step_override = openearthagent(regime="step", vision=False, task_id="0", max_iterations=9)
    assert step_override.metadata["max_iterations"] == 9
    assert step_override.solver.__registry_info__.name == "inspect_ai/generate"


@pytest.mark.parametrize(
    "build",
    [
        lambda: earth_bench(max_iterations=0),
        lambda: geobenchx(max_iterations=0),
        lambda: eo_gym(vision=False, max_iterations=0),
        lambda: openearthagent(regime="step", vision=False, task_id="0", max_iterations=0),
    ],
)
def test_max_iterations_below_one_is_rejected(build):
    with pytest.raises(ValueError, match="at least 1"):
        build()
