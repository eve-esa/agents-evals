"""Inspect scorer wiring for OpenEarthAgent. The judge is a stub."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from agentic_evals.agents.trace_callback import TRACE_KEY
from agentic_evals.tasks.openearthagent.scorers import (
    NOT_APPLICABLE_KEY,
    openearthagent_scorer,
    openearthagent_step,
)


class _Judge:
    def __init__(self, text: str = '{"Score": 0.5, "Justification": "close"}', fail: bool = False):
        self.text = text
        self.fail = fail
        self.prompts: list[str] = []

    def __str__(self) -> str:
        return "stub-judge"

    async def generate(self, messages, config=None):
        del config
        self.prompts.append(messages[0].content)
        if self.fail:
            raise RuntimeError("judge down")
        return SimpleNamespace(completion=self.text)


def _state(**kwargs) -> TaskState:
    defaults = dict(
        model="test",
        sample_id="0",
        epoch=1,
        input="q",
        messages=[ChatMessageUser(content="How many?")],
        metadata={"question": "How many?", "reference_answer": "2", "is_generation": False, "gold_names": ["Calculator", "Terminate"]},
        output=ModelOutput.from_content(model="test", content="2"),
    )
    defaults.update(kwargs)
    return TaskState(**defaults)


def _score(factory, state: TaskState):
    return asyncio.run(factory()(state, Target("")))


def test_trace_beats_messages_and_a_failed_call_is_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: _Judge())
    state = _state()
    state.store.set(TRACE_KEY, [{"name": "Calculator", "input": '{"expression": "1+1"}', "error": True}])
    state.messages.append(
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="1", function="OCR", arguments={"image": "img_1"})],
        )
    )
    score = _score(openearthagent_scorer, state)
    assert score.metadata["call_source"] == "trace"
    assert score.metadata["pred_names"] == ["Calculator"]
    assert score.value["any_order"] == 1.0
    assert score.value["answer"] == 0.5


def test_messages_are_the_trace_when_the_store_is_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: _Judge())
    state = _state(
        messages=[
            ChatMessageUser(content="How many?"),
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id="1", function="Calculator", arguments={"expression": "1+1"})],
            ),
        ]
    )
    score = _score(openearthagent_scorer, state)
    assert score.metadata["call_source"] == "messages"
    assert score.metadata["pred_names"] == ["Calculator"]
    assert score.value["any_order"] == 1.0


def test_terminate_json_is_judged_on_ans(monkeypatch: pytest.MonkeyPatch):
    judge = _Judge()
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: judge)
    text = '{"actions": [{"name": "Terminate", "arguments": {"ans": "12 m"}}]}'
    state = _state(output=ModelOutput.from_content(model="test", content=text))
    score = _score(openearthagent_scorer, state)
    assert score.answer == "12 m"
    assert score.value["answer"] == 0.5
    assert "12 m" in judge.prompts[0]
    assert "Terminate" not in judge.prompts[0].split("Predicted Answer:", 1)[1]


def test_plain_text_is_the_answer_and_an_empty_one_skips_the_judge(monkeypatch: pytest.MonkeyPatch):
    judge = _Judge()
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: judge)
    filled = _score(openearthagent_scorer, _state())
    assert filled.answer == "2"
    assert filled.value["answer"] == 0.5
    empty = _score(
        openearthagent_scorer,
        _state(output=ModelOutput.from_content(model="test", content="")),
    )
    assert empty.value["answer"] == 0.0
    assert len(judge.prompts) == 1


def test_generation_task_scores_the_last_tool_and_skips_the_judge(monkeypatch: pytest.MonkeyPatch):
    judge = _Judge()
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: judge)
    state = _state(
        metadata={
            "question": "Draw it.",
            "reference_answer": "map",
            "is_generation": True,
            "gold_names": ["DrawBox", "Terminate"],
        }
    )
    state.store.set(TRACE_KEY, [{"name": "DrawBox", "input": {"image": "img_1"}, "error": False}])
    score = _score(openearthagent_scorer, state)
    assert score.value["gen_success"] == 1.0
    assert math.isnan(score.value["answer"])
    assert judge.prompts == []


def test_vision_skip_and_a_failed_judge_are_nan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "agentic_evals.tasks.openearthagent.scorers.get_model",
        lambda *args, **kwargs: _Judge(fail=True),
    )
    skipped = _state()
    skipped.store.set(NOT_APPLICABLE_KEY, "no vision")
    skipped_score = _score(openearthagent_scorer, skipped)
    assert math.isnan(skipped_score.value)
    failed = _score(openearthagent_scorer, _state())
    assert math.isnan(failed.value["answer"])
    assert "RuntimeError" in failed.explanation


def test_step_planning_turn_is_exempt_and_terminate_is_judged(monkeypatch: pytest.MonkeyPatch):
    judge = _Judge()
    monkeypatch.setattr("agentic_evals.tasks.openearthagent.scorers.get_model", lambda *args, **kwargs: judge)
    planning = _state(
        metadata={"gold_text": '{"thought": "plan", "actions": []}', "is_generation": False, "question": "How many?"},
        output=ModelOutput.from_content(model="test", content='{"actions": []}'),
    )
    exempt = _score(openearthagent_step, planning)
    assert math.isnan(exempt.value)
    assert exempt.metadata["exempt"] is True
    gold = '{"actions": [{"name": "Terminate", "arguments": {"ans": "2"}}]}'
    answered = _state(
        metadata={"gold_text": gold, "is_generation": False, "question": "How many?"},
        output=ModelOutput.from_content(model="test", content=gold),
    )
    score = _score(openearthagent_step, answered)
    assert score.value["tool"] == 1.0
    assert score.value["answer"] == 0.5
    assert judge.prompts
