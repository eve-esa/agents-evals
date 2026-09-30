"""Capture what the model endpoint says it is, at run time.

AGENTS.md: the system under test is deployed on demand, so its identity is read
from the endpoint when the run starts, never assumed from configuration. For
OpenAI-compatible endpoints this is ``GET {base_url}/models``. If the endpoint
does not list the requested model, or lists no version/owner information, the
record says so (``identified: false``) rather than filling in a plausible value.

The record is stored per sample (``RUN_IDENTITY_KEY`` in the sample store, so it
is persisted in the ``.eval`` log) and fetched once per endpoint per process.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

RUN_IDENTITY_KEY = "agentic_evals:run_identity"

_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _api_key(model_api: Any) -> str | None:
    if getattr(model_api, "api_key", None):
        return model_api.api_key
    for var in getattr(model_api, "api_key_vars", []) or []:
        if os.environ.get(var):
            return os.environ[var]
    return None


def fetch_identity(model_name: str, base_url: str | None, api_key: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model": model_name,
        "base_url": base_url,
        "queried_at": time.time(),
        "identified": False,
        "endpoint_model": None,
        "error": None,
    }
    if not base_url:
        record["error"] = "no base_url for this provider; identity not queried"
        return record
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=15)
        response.raise_for_status()
        listed = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001 - the failure itself is the record
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    if not isinstance(listed, list):
        record["error"] = f"/models data is {type(listed).__name__}, expected a list"
        return record
    match = match_listed_model(model_name, listed)
    if match is None:
        record["error"] = f"endpoint does not list {model_name!r} ({len(listed)} models listed)"
        return record
    record["endpoint_model"] = match
    record["identified"] = True
    return record


def match_listed_model(model_name: str, listed: list[Any]) -> dict[str, Any] | None:
    """The listed model whose id is ``model_name`` or a slash-bounded suffix of it.

    OpenAI-compatible ``/models`` ids are sometimes the bare checkpoint
    (``gpt-4o``) and sometimes ``author/name``. OpenRouter's list uses the
    latter (``qwen/qwen3-32b``; see
    https://openrouter.ai/docs/api/api-reference/models/get-a-model-by-its-slug).
    Inspect's ``openai-api`` model name adds the gateway in front
    (``openrouter/qwen/qwen3-32b``). The longest matching id wins, so
    ``qwen/qwen3-32b`` is preferred over a bare ``qwen3-32b`` when both exist.
    """
    candidates: list[dict[str, Any]] = []
    for entry in listed:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if model_id == model_name or model_name.endswith("/" + model_id):
            candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda entry: len(entry["id"]))


@solver
def record_run_identity() -> Solver:
    """Setup solver: store the endpoint's self-reported identity for the model under test."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        base_url = getattr(model.api, "base_url", None)
        key = (str(model), str(base_url))
        if key not in _cache:
            _cache[key] = fetch_identity(model.name, base_url, _api_key(model.api))
        state.store.set(RUN_IDENTITY_KEY, _cache[key])
        return state

    return solve
