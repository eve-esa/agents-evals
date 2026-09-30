"""``--max-tokens -1`` is recorded on the eval and omitted from the request."""

from inspect_ai._util.registry import registry_find
from inspect_ai.model import GenerateConfig
from inspect_ai.model._openai import openai_completion_params

import agentic_evals._registry  # noqa: F401  — registers the hook
from agentic_evals.max_tokens import UNCAPPED_MAX_TOKENS, apply_uncapped_max_tokens


def test_minus_one_is_omitted_from_the_provider_request():
    config = GenerateConfig(max_tokens=UNCAPPED_MAX_TOKENS, temperature=0)
    apply_uncapped_max_tokens(config)
    assert config.max_tokens is None
    assert config.temperature == 0
    params = openai_completion_params("mistralai/mistral-small", config, tools=False)
    assert "max_tokens" not in params


def test_a_positive_cap_is_sent():
    config = GenerateConfig(max_tokens=4096)
    apply_uncapped_max_tokens(config)
    assert config.max_tokens == 4096
    params = openai_completion_params("mistralai/mistral-small", config, tools=False)
    assert params["max_tokens"] == 4096


def test_uncapped_hook_is_registered():
    found = registry_find(
        lambda info: info.type == "hooks" and info.name.endswith("/uncapped_max_tokens")
    )
    assert len(found) == 1
