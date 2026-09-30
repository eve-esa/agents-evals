"""Endpoint identity must use the id the server lists, including author/name ids."""

from agentic_evals.run_identity import match_listed_model

# OpenRouter lists author/name. Inspect's openai-api model name prefixes the gateway.
# https://openrouter.ai/docs/api/api-reference/models/get-a-model-by-its-slug
OPENROUTER = [
    {"id": "qwen/qwen3-32b", "canonical_slug": "qwen/qwen3-32b"},
    {"id": "deepseek/deepseek-chat-v3.1"},
    {"id": "openai/gpt-4o"},
]


def test_openrouter_author_name_matches_inspect_model_name():
    match = match_listed_model("openrouter/qwen/qwen3-32b", OPENROUTER)
    assert match is not None
    assert match["id"] == "qwen/qwen3-32b"


def test_exact_id_matches():
    match = match_listed_model("qwen/qwen3-32b", OPENROUTER)
    assert match is not None
    assert match["id"] == "qwen/qwen3-32b"


def test_bare_checkpoint_id_matches():
    # A self-hosted /v1/models list often uses the checkpoint name alone.
    listed = [{"id": "eve-instruct", "owned_by": "eve"}]
    match = match_listed_model("eve/eve-instruct", listed)
    assert match is not None
    assert match["owned_by"] == "eve"


def test_longest_id_wins_when_a_shorter_suffix_is_also_listed():
    listed = [{"id": "qwen3-32b"}, {"id": "qwen/qwen3-32b", "name": "Qwen3 32B"}]
    match = match_listed_model("openrouter/qwen/qwen3-32b", listed)
    assert match is not None
    assert match["id"] == "qwen/qwen3-32b"
    assert match["name"] == "Qwen3 32B"


def test_unrelated_models_do_not_match():
    assert match_listed_model("openrouter/qwen/qwen3-32b", [{"id": "qwen/qwen3-14b"}]) is None
    assert match_listed_model("gpt-4o-mini", [{"id": "gpt-4o"}]) is None
