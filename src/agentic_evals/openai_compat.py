"""Work around Inspect 0.3.268 crashing on an empty chat completion.

The OpenAI SDK builds a ``ChatCompletion`` with ``construct_type``, which
accepts an HTTP 200 body whose ``choices`` field is null or absent. OpenRouter
uses that shape when the upstream provider fails
(https://openrouter.ai/docs/api/reference/errors). Inspect then executes
``list(response.choices)`` in ``chat_choices_from_openai`` and the sample dies
with ``TypeError``.

A 5xx body is raised as ``OpenAIResponseError(code="server_error")``, which
``openai_classify_retry`` already treats as transient. A 4xx body (the context
window exceeded, a bad schema) is raised as ``invalid_request`` and is not
retried: retrying it only adds silent backoff.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model._openai import OpenAIResponseError, chat_choices_from_openai

_original = chat_choices_from_openai


def _error_code(detail: Any) -> str:
    status = detail.get("code") if isinstance(detail, dict) else None
    if isinstance(status, int) and 400 <= status < 500:
        return "invalid_request"
    return "server_error"


def _empty_choices_error(response: Any) -> OpenAIResponseError:
    extra = getattr(response, "model_extra", None) or {}
    detail = extra.get("error", extra) if isinstance(extra, dict) else extra
    return OpenAIResponseError(
        _error_code(detail),
        f"provider returned a chat completion with no choices ({detail!r})",
    )


def _guarded_chat_choices_from_openai(
    response: Any,
    tools: Any,
    reasoning_extractor: Any = None,
) -> Any:
    if not getattr(response, "choices", None):
        raise _empty_choices_error(response)
    return _original(response, tools, reasoning_extractor)


def install() -> None:
    """Replace Inspect's choice parser in both modules that hold a reference."""
    import inspect_ai.model._openai as openai_mod
    import inspect_ai.model._providers.openai_compatible as compatible_mod

    openai_mod.chat_choices_from_openai = _guarded_chat_choices_from_openai
    compatible_mod.chat_choices_from_openai = _guarded_chat_choices_from_openai
