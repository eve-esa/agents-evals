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


def test_ordinary_text_is_left_unchanged():
    message = AIMessage(content="The answer is 4.")
    assert normalize_mistral_message(message) is message
    assert mistral_tool_calls_from_text("not a call") == []


def test_openrouter_ignore_is_the_provider_body():
    assert openrouter_extra_body("") is None
    assert openrouter_extra_body("deepinfra, venice") == {
        "provider": {"ignore": ["deepinfra", "venice"], "allow_fallbacks": True}
    }
