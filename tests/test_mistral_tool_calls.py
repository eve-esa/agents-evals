"""Mistral tool-call normalisation and OpenRouter provider routing."""

from langchain_core.messages import AIMessage

from agentic_evals.agents.langgraph_react import (
    mistral_tool_calls_from_text,
    normalize_mistral_message,
)
from agentic_evals.tasks.geobenchx.task import openrouter_extra_body


def test_whitespace_on_a_structured_tool_name_is_stripped():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "reject_task\n ",
                "args": {"reason": "unsolvable"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    normalized = normalize_mistral_message(message)
    assert normalized.tool_calls[0]["name"] == "reject_task"
    assert normalized.tool_calls[0]["id"] == "call_1"
    assert normalized.tool_calls[0]["args"] == {"reason": "unsolvable"}


def test_invalid_tool_call_with_only_whitespace_in_the_name_is_promoted():
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call_2",
                "name": "reject_task\n",
                "args": '{"reason": "no"}',
                "error": "name",
            }
        ],
    )
    normalized = normalize_mistral_message(message)
    assert normalized.tool_calls[0]["name"] == "reject_task"
    assert normalized.tool_calls[0]["args"] == {"reason": "no"}
    assert normalized.invalid_tool_calls == []


def test_mistral_v11_envelope_becomes_a_tool_call():
    text = '[TOOL_CALLS]reject_task[ARGS]{"reason": "unsolvable"}'
    calls = mistral_tool_calls_from_text(text)
    assert calls == [
        {
            "name": "reject_task",
            "args": {"reason": "unsolvable"},
            "id": "call_mistral_0",
            "type": "tool_call",
        }
    ]
    message = normalize_mistral_message(AIMessage(content="I should stop.\n" + text))
    assert message.content == "I should stop."
    assert message.tool_calls[0]["name"] == "reject_task"


def test_mistral_json_array_envelope_becomes_tool_calls():
    text = '[TOOL_CALLS][{"name": "get_filelist", "arguments": {"dir_path": "d"}}]'
    calls = mistral_tool_calls_from_text(text)
    assert calls[0]["name"] == "get_filelist"
    assert calls[0]["args"] == {"dir_path": "d"}


def test_empty_tool_name_is_dropped():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "", "args": {}, "id": "call_empty", "type": "tool_call"}],
        additional_kwargs={
            "tool_calls": [
                {"id": "call_empty", "type": "function", "function": {"name": "", "arguments": "{}"}}
            ]
        },
    )
    normalized = normalize_mistral_message(message)
    assert normalized.tool_calls == []
    assert "tool_calls" not in normalized.additional_kwargs


def test_empty_tool_name_is_dropped_beside_a_real_call():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "", "args": {}, "id": "call_empty", "type": "tool_call"},
            {
                "name": "GetAreaBoundary",
                "args": {"area": "Banff"},
                "id": "call_ok",
                "type": "tool_call",
            },
        ],
    )
    normalized = normalize_mistral_message(message)
    assert [call["name"] for call in normalized.tool_calls] == ["GetAreaBoundary"]
    assert normalized.tool_calls[0]["id"] == "call_ok"


def test_empty_structured_call_falls_through_to_the_text_envelope():
    text = '[TOOL_CALLS]GetAreaBoundary[ARGS]{"area": "Banff"}'
    message = AIMessage(
        content=text,
        tool_calls=[{"name": "", "args": {}, "id": "call_empty", "type": "tool_call"}],
    )
    normalized = normalize_mistral_message(message)
    assert normalized.content == ""
    assert normalized.tool_calls[0]["name"] == "GetAreaBoundary"
    assert normalized.tool_calls[0]["args"] == {"area": "Banff"}


def test_invalid_call_with_an_empty_name_is_not_replayed():
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call_blank",
                "name": "",
                "args": "{}",
                "error": "name",
            }
        ],
    )
    normalized = normalize_mistral_message(message)
    assert normalized.tool_calls == []
    assert normalized.invalid_tool_calls == []


def test_ordinary_text_is_left_unchanged():
    message = AIMessage(content="The answer is 4.")
    assert normalize_mistral_message(message) is message
    assert mistral_tool_calls_from_text("not a call") == []


def test_openrouter_ignore_is_the_provider_body():
    assert openrouter_extra_body("") is None
    assert openrouter_extra_body("deepinfra, venice") == {
        "provider": {"ignore": ["deepinfra", "venice"], "allow_fallbacks": True}
    }
    assert openrouter_extra_body(only="Venice") == {
        "provider": {"only": ["venice"], "allow_fallbacks": False}
    }
    assert openrouter_extra_body("parasail", only="venice")["provider"]["only"] == ["venice"]
