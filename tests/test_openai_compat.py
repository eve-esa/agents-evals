"""Empty OpenRouter completions must be retried, not raised as TypeError."""

from openai._models import construct_type
from openai.types.chat import ChatCompletion

from agentic_evals.openai_compat import install
from inspect_ai.model._openai import OpenAIResponseError, openai_classify_retry
from inspect_ai.model._providers import openai_compatible


def test_null_choices_is_a_transient_server_error() -> None:
    install()
    completion = construct_type(
        type_=ChatCompletion,
        value={"error": {"message": "Provider returned error", "code": 502}},
    )
    assert completion.choices is None

    try:
        openai_compatible.chat_choices_from_openai(completion, [])
    except OpenAIResponseError as exc:
        decision = openai_classify_retry(exc)
        assert exc.code == "server_error"
        assert decision is not None
        assert decision.kind == "transient"
    else:
        raise AssertionError("expected OpenAIResponseError")


def test_context_length_error_is_not_retried() -> None:
    install()
    completion = construct_type(
        type_=ChatCompletion,
        value={
            "error": {
                "message": "maximum context length is 40960 tokens",
                "code": 400,
            }
        },
    )

    try:
        openai_compatible.chat_choices_from_openai(completion, [])
    except OpenAIResponseError as exc:
        assert exc.code == "invalid_request"
        assert openai_classify_retry(exc) is None
    else:
        raise AssertionError("expected OpenAIResponseError")
