"""Work around Inspect 0.3.268 crashing on an empty chat completion, and stop retrying a spent quota.

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

``openai_classify_retry`` retries every HTTP 429 as a rate limit. OpenAI marks
a spent quota with ``insufficient_quota`` on the same status
(https://platform.openai.com/docs/guides/error-codes); the EVE endpoint sends
that type with code ``token_budget_exceeded`` and a reset date weeks away.
Those 429s are not retried, so the sample fails at once with the provider's
message instead of backing off until ``max_retries``.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model._openai import OpenAIResponseError, chat_choices_from_openai, openai_classify_retry

_original = chat_choices_from_openai
_original_classify = openai_classify_retry

QUOTA_CODES = frozenset({"insufficient_quota", "token_budget_exceeded"})


def is_quota_exhausted(ex: BaseException) -> bool:
    """True for a 429 whose body says the quota is spent, not that requests came too fast."""
    if getattr(ex, "status_code", None) != 429:
        return False
    body = getattr(ex, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    return bool(QUOTA_CODES & {error.get("code"), error.get("type")})


def _classify_retry(ex: BaseException) -> Any:
    if is_quota_exhausted(ex):
        return None
    return _original_classify(ex)


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
    """Replace Inspect's choice parser and retry classifier in every module that holds a reference."""
    import inspect_ai.model._openai as openai_mod
    import inspect_ai.model._providers.openai as openai_provider_mod
    import inspect_ai.model._providers.openai_compatible as compatible_mod

    openai_mod.chat_choices_from_openai = _guarded_chat_choices_from_openai
    compatible_mod.chat_choices_from_openai = _guarded_chat_choices_from_openai
    for module in (openai_mod, openai_provider_mod, compatible_mod):
        module.openai_classify_retry = _classify_retry
