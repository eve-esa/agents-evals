"""``--max-tokens -1`` sends no output cap.

Inspect's ``--max-tokens`` is an integer
(``GenerateConfig.max_tokens``; the CLI option is ``type=int``). ``-1`` is
stored on the eval config, so a run that chose not to cap is distinct from
one that left the default unset, and this hook removes it before the
provider call.

Omitting the field is the OpenRouter behaviour: a sampling parameter that is
absent is not replaced with a hardcoded value, and the endpoint applies its
own maximum, at most the context length minus the prompt. The model still
stops when it is finished. A positive ``--max-tokens`` remains a hard cap.

https://openrouter.ai/docs/api-reference/parameters
https://inspect.aisi.org.uk/options.html
"""

from __future__ import annotations

from inspect_ai.hooks import BeforeModelGenerate, Hooks, hooks
from inspect_ai.model import GenerateConfig

# Sentinel on the eval config. Not sent to the provider.
UNCAPPED_MAX_TOKENS = -1


def apply_uncapped_max_tokens(config: GenerateConfig) -> None:
    """Drop the sentinel so the request does not include ``max_tokens``."""
    if config.max_tokens == UNCAPPED_MAX_TOKENS:
        config.max_tokens = None


@hooks(
    name="uncapped_max_tokens",
    description="Treat --max-tokens -1 as no output cap.",
)
class UncappedMaxTokens(Hooks):
    async def on_before_model_generate(self, data: BeforeModelGenerate) -> None:
        apply_uncapped_max_tokens(data.config)
