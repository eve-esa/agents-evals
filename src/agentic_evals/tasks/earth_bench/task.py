"""Earth-Bench (Feng et al., ICLR 2026, arXiv:2509.23141) as an Inspect task.

Run with, e.g.::

    uv run inspect eval agentic_evals/earth_bench -T regime=ap -T modality=spectrum \\
        --model openai-api/<name>/<model> -M strict_tools=false

``-M strict_tools=false`` is needed with Inspect's ``openai-api`` provider, which
marks tools ``strict`` by default: several upstream tools declare untyped
arguments, which OpenAI's strict mode rejects. This is a provider setting, not a
change to the benchmark's tool schemas.

The default solver is our LangGraph ReAct agent; any other agent can be passed
with ``--solver``. The setup solvers run for every solver: they record the
endpoint's identity, leave RGB samples unscored when Perception's lookup file
is absent, and publish this sample's MCP tool servers.
"""

from __future__ import annotations

import tomllib

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agentic_evals.agents.langgraph_react import langgraph_react
from agentic_evals.paths import REPO_ROOT
from agentic_evals.run_identity import record_run_identity
from agentic_evals.scorers.judges import extra_judges
from agentic_evals.tasks.earth_bench.dataset import AnswerKey, Modality, Regime, earth_bench_dataset
from agentic_evals.tasks.earth_bench.scorers import NOT_APPLICABLE_KEY, earth_bench_scorer
from agentic_evals.tasks.earth_bench.tools import rgb_unavailable_reason, sample_servers
from agentic_evals.tooling import publish_mcp_servers


@solver
def skip_unavailable_rgb() -> Solver:
    """Leave RGB samples unscored when Perception's lookup file is absent.

    Sets ``completed`` so the agent and its MCP servers are not started.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reason = rgb_unavailable_reason(str(state.metadata.get("modality", "")))
        if reason is not None:
            state.store.set(NOT_APPLICABLE_KEY, reason)
            state.completed = True
        return state

    return solve


def _upstream_revisions() -> dict[str, str]:
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    return {
        "earth_agent_rev": lock["earth_agent"]["rev"],
        "earth_bench_data_rev": lock["earth_bench_data"]["revision"],
    }


@task
def earth_bench(
    regime: Regime = "ap",
    modality: Modality = "spectrum",
    answer_key: AnswerKey = "gt_file",
    upstream_compat: bool = False,
    trajectory_judges: str | list[str] = "",
    solver: Solver | Agent | None = None,
) -> Task:
    """Earth-Bench tasks with Earth-Agent's MCP tool kits.

    Args:
        regime: "ap" (Auto-Planning) or "if" (Instruction-Following); report separately.
        modality: "spectrum", "products", "rgb" or "all". RGB samples are left
            unscored when Perception's lookup CSV is not on this machine.
        answer_key: "gt_file" (what upstream's published accuracies use) or "question_json".
        upstream_compat: Reproduce upstream metric quirks, including scoring against
            the published gold with no scalar expansion (only needed to match published files).
        trajectory_judges: Comma-separated judge models for optional whole-trajectory LLM judges
            (our addition, one scorer per judge; see agentic_evals.scorers.judges).
        solver: Agent to evaluate (default: LangGraph ReAct, upstream's recursion limit 50).
    """
    return Task(
        dataset=earth_bench_dataset(regime, modality, answer_key),
        setup=[record_run_identity(), skip_unavailable_rgb(), publish_mcp_servers(sample_servers)],
        solver=solver or langgraph_react(),
        scorer=[earth_bench_scorer(upstream_compat=upstream_compat), *extra_judges(trajectory_judges)],
        # Upstream runners use temperature 0.1; --temperature overrides.
        config=GenerateConfig(temperature=0.1),
        metadata={
            "benchmark": "earth_bench",
            "regime": regime,
            "modality": modality,
            "answer_key": answer_key,
            "upstream_compat": upstream_compat,
            "trajectory_judges": trajectory_judges,
            **_upstream_revisions(),
        },
        version=1,
    )
