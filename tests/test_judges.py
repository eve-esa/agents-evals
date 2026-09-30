from __future__ import annotations

import pytest

from agentic_evals.scorers.agreement import cohen_kappa, pairwise
from agentic_evals.scorers.judges import _as_openai_messages, extra_judges


def test_cohen_kappa_known_values():
    # Perfect agreement, chance-level agreement, and a textbook 2x2 case:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> p_o=0.7, p_e=0.5, kappa=0.4.
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_pairwise_uses_common_samples():
    table = {"x": {"1": True, "2": False, "3": True}, "y": {"1": True, "2": True}}
    (row,) = pairwise(table)
    assert row["n"] == 2 and row["agreement"] == 0.5


def test_reference_rendering_and_factory():
    msgs = _as_openai_messages([{"name": "get_filelist", "arguments": {"dir_path": "d"}, "output": ["a"]}], "B")
    assert msgs[0]["tool_calls"][0]["function"] == {"name": "get_filelist", "arguments": '{"dir_path": "d"}'}
    assert msgs[1] == {"role": "tool", "tool_call_id": "call_0", "content": '["a"]'}
    assert msgs[-1] == {"role": "assistant", "content": "B"}
    assert extra_judges("") == [] and len(extra_judges("openai/a, openai/b")) == 2
