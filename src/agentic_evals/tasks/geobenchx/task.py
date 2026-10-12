"""GeoBenchX (Krechetova & Kochedykov, SIGSPATIAL 2025, arXiv:2503.18129) as an Inspect task.

Run with, e.g.::

    uv run inspect eval agentic_evals/geobenchx -T group=spatial \\
        --model openai-api/<name>/<model> -M strict_tools=false \\
        --max-tokens -1 \\
        --model-role grader=openai/gpt-4.1-2025-04-14

``--max-tokens -1`` records that this run does not impose an output cap and
omits ``max_tokens`` on the provider request (see
``agentic_evals.max_tokens``). A positive value is a hard cap. The model
still stops when it is done; the endpoint's own context window remains.

``-M strict_tools=false`` is needed with Inspect's ``openai-api`` provider:
several tools take a dict (categorical filters) or a free-form list, which
OpenAI's strict mode rejects.

The agent is our LangGraph ReAct agent. Upstream's ``execute_task`` uses
temperature 0 and ``max_steps=25``, passed as LangGraph's ``recursion_limit``
(graph supersteps; their paper calls the same parameter "25 iterations").
Override that budget with ``-T max_iterations=`` or ``task.args.max_iterations``.
The system prompt is their
``SYSTEM_PROMPT + RULES_PROMPT``, including today's date. Agent generates use
the same idle timeout and retry cap as the match judge
(``GenerateConfig.stream_idle_timeout``): silence expires an attempt, a stream
that is still producing tokens does not. ``openrouter_ignore`` is OpenRouter's
``provider.ignore`` (https://openrouter.ai/docs/guides/routing/provider-selection)
and applies only to the agent model. Mistral Small 3.2 on that router should
be pinned to ``venice`` (``-T openrouter_only=venice``): it is the only host
whose context is 256k. Parasail is 131072 and rejects a prompt that long.
Venice and DeepInfra return HTTP 400 when a tool name is ``reject_task`` plus
trailing whitespace; the agent strips that whitespace before the next call.

The official scores are the shares of no match, partial, and match
(Krechetova & Kochedykov, Section 6.1; ``generate_eval_stats``) and
``length_diff``. Each judge is a separate result row named with its
checkpoint id, without the ``openai-api`` provider prefix. The panel share
is the unweighted mean of those shares (Section 5.2), reported by
``geobenchx_panel``. ``geobenchx_merged`` averages the indicators on one
sample and is not that panel number. ``geobenchx_steps`` (TAO, TIO, TEM,
parameter accuracy) is reported alongside and is not part of their protocol.
The paper's judge panel is Claude Sonnet 3.5 (``claude-3-5-sonnet-20241022``),
GPT-4.1 (``gpt-4.1-2025-04-14``, the model that scored
``evaluator_tuning_set.json``) and Gemini 2.5 Pro Preview
(``gemini-2.5-pro-preview-05-06``). Pass them with ``-T judges=...``; each is
stored separately. A retired checkpoint is not replaced.
"""

from __future__ import annotations

import tomllib

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver

from agentic_evals.agents.langgraph_react import langgraph_react
from agentic_evals.paths import REPO_ROOT
from agentic_evals.run_identity import record_run_identity
from agentic_evals.scorers.judges import extra_judges
from agentic_evals.tasks.geobenchx.dataset import Group, Kind, geobenchx_dataset
from agentic_evals.tasks.geobenchx.scorers import (
    JUDGE_MAX_RETRIES,
    JUDGE_STREAM_IDLE_TIMEOUT,
    extra_match_judges,
    geobenchx_merged,
    geobenchx_panel,
    geobenchx_scorer,
    geobenchx_steps,
)
from agentic_evals.tasks.geobenchx.tools import ported_code_sha256, sample_servers
from agentic_evals.tooling import publish_mcp_servers

def _revision() -> str:
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    return lock["geobenchx"]["rev"]


def _provider_names(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return [part.strip().lower() for part in value if part.strip()]


def openrouter_extra_body(
    ignore: str | list[str] = "",
    only: str | list[str] = "",
) -> dict[str, object] | None:
    """OpenRouter provider routing, or ``None`` when routing is left alone.

    Slugs are the lowercase provider names from the endpoints API
    (``deepinfra``, ``venice``, ``parasail``, ``mistral``).
    https://openrouter.ai/docs/guides/routing/provider-selection

    ``only`` pins those hosts and disables fallbacks. ``ignore`` skips hosts
    and still allows the rest. ``only`` wins when both are set.
    """
    pinned = _provider_names(only)
    if pinned:
        return {"provider": {"only": pinned, "allow_fallbacks": False}}
    skipped = _provider_names(ignore)
    if not skipped:
        return None
    return {"provider": {"ignore": skipped, "allow_fallbacks": True}}


@task
def geobenchx(
    group: Group = "all",
    kind: Kind = "all",
    task_id: str | None = None,
    judges: str | list[str] = "",
    trajectory_judges: str | list[str] = "",
    openrouter_ignore: str | list[str] = "",
    openrouter_only: str | list[str] = "",
    max_iterations: int = 25,
    solver: Solver | Agent | None = None,
) -> Task:
    """GeoBenchX tasks with GeoBenchX's tools over MCP.

    Args:
        group: "merge", "process", "spatial", "heatmaps", or "all".
        kind: "solvable", "unsolvable", "control", or "all". The paper's
            solvable set is our solvable plus control.
        task_id: Restrict to one ``task_ID``.
        judges: Extra match-score judges, comma-separated. The ``grader`` role
            is always one judge, separate from these.
        trajectory_judges: Optional agentevals whole-trajectory judges (our
            addition; see ``agentic_evals.scorers.judges``).
        openrouter_ignore: Comma-separated OpenRouter provider slugs the agent
            must not use. Empty leaves routing alone.
        openrouter_only: Comma-separated OpenRouter provider slugs the agent
            must use, with fallbacks off. For Mistral Small 3.2 this is
            ``venice``: it is the only host with a 256k context. Parasail is
            131072, DeepInfra 128000, and Mistral's own endpoint 32768.
        max_iterations: LangGraph superstep cap. Upstream names this
            ``max_steps`` (default 25) and passes it as ``recursion_limit``.
            A YAML run config overrides it with ``task.args.max_iterations``.
        solver: Agent to evaluate (default: LangGraph ReAct). A custom solver
            keeps its own limit; this argument is still recorded.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    dataset = geobenchx_dataset(group, kind, task_id)
    prompt_date = dataset[0].metadata["prompt_date"] if len(dataset) else ""
    provider = openrouter_extra_body(openrouter_ignore, openrouter_only)
    return Task(
        dataset=dataset,
        setup=[record_run_identity(), publish_mcp_servers(sample_servers)],
        solver=solver or langgraph_react(recursion_limit=max_iterations),
        scorer=[
            geobenchx_steps(),
            geobenchx_scorer(),
            *extra_match_judges(judges),
            geobenchx_merged(),
            geobenchx_panel(),
            *extra_judges(trajectory_judges),
        ],
        # Inspect view's summary shows the headline scorer's rows, three of
        # them (ResultsPanel, kMaxPrimaryScoreRows). Those rows are the
        # panel shares, in the paper's table order.
        headline_metric="geobenchx_panel.match",
        config=GenerateConfig(
            temperature=0,
            max_retries=JUDGE_MAX_RETRIES,
            stream_idle_timeout=JUDGE_STREAM_IDLE_TIMEOUT,
            extra_body=provider,
        ),
        metadata={
            "benchmark": "geobenchx",
            "summary_groups": ["kind", "labels"],
            "group": group,
            "kind": kind,
            "task_id": task_id,
            "judges": judges,
            "trajectory_judges": trajectory_judges,
            "openrouter_ignore": openrouter_ignore,
            "openrouter_only": openrouter_only,
            "max_iterations": max_iterations,
            "prompt_date": prompt_date,
            "geobenchx_rev": _revision(),
            "tools_sha256": ported_code_sha256(),
        },
        version=1,
    )
