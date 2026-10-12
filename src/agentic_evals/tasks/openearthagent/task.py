"""OpenEarthAgent (Shabbir et al., arXiv:2602.17665) as an Inspect task.

Two regimes, reported separately:

- ``step`` — teacher-forced. Each gold turn is one sample. The model sees
  ``RS_AGENT_PROMPT`` and the gold history and writes the next JSON action.
  No tools run, so this needs no GPU. Upstream's ``max_length`` of 256 new
  tokens and the OpenAI driver's temperature 0.4 are the defaults.
- ``e2e`` — live tools through the MCP proxy. A local controller that is
  not already up is started from the cloned tool server (Conda
  ``tool-server-e1`` / ``tool-server-e2``) and stopped when the eval
  process exits. ``OPENEARTHAGENT_CONTROLLER_URL`` defaults to
  ``http://127.0.0.1:20001``. ``max_iterations`` 15 is the model-turn cap
  (upstream ``--max_rounds``). Step mode records the same number and still
  generates once per gold turn.
  The five gpkg tools return a cache hit; every miss is sent to the
  live worker.

Image tasks are marked not applicable when ``vision=false`` (a text-only
model). They are not given a score. ``gpt-oss-20b`` is text-only, so it is
not applicable on the 533 image tasks.

The official judge is the ``grader`` role, ``gpt-4o-mini`` in the paper::

    uv run inspect eval agentic_evals/openearthagent -T regime=step \\
        --model openai-api/<name>/<model> \\
        --model-role grader=openai/gpt-4o-mini

    uv run inspect eval agentic_evals/openearthagent -T regime=e2e \\
        --model openai-api/<name>/<vlm> -M strict_tools=false \\
        --model-role grader=openai/gpt-4o-mini

Logged rates are fractions in ``[0, 1]``. Tables 3 and 4 print percentages.
"""

from __future__ import annotations

import asyncio
import tomllib

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from agentic_evals.agents.langgraph_react import langgraph_react
from agentic_evals.paths import REPO_ROOT
from agentic_evals.run_identity import record_run_identity
from agentic_evals.tasks.openearthagent.dataset import Category, Regime, openearthagent_dataset
from agentic_evals.tasks.openearthagent.scorers import (
    NOT_APPLICABLE_KEY,
    extra_answer_judges,
    openearthagent_scorer,
    openearthagent_step,
)
from agentic_evals.tasks.openearthagent.tools import (
    TOOL_SERVICE_KEY,
    controller_url,
    ported_code_sha256,
    sample_servers_factory,
)
from agentic_evals.tasks.openearthagent.tools.server import ensure_controller
from agentic_evals.tooling import publish_mcp_servers

# OpenAIModels defaults to 0.4. config.yaml generation_config.max_length is 256
# new tokens on that driver. eval_e2e.sh and eval_step.sh both pass
# --max_rounds 15. batch_inference stops there. step_inference does not read it.
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 256
DEFAULT_MAX_ITERATIONS = 15


def _revision() -> dict[str, str]:
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    spec = lock["openearthagent"]
    return {"rev": spec["rev"], "dataset_revision": spec["revision"]}


@solver
def skip_without_vision() -> Solver:
    """Mark image tasks not applicable and do not run the agent on them."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata.get("needs_vision"):
            state.store.set(
                NOT_APPLICABLE_KEY,
                "OpenEarthAgent image tasks need a vision-capable model; this run set vision=false",
            )
            state.completed = True
        return state

    return solve


def prepare_tool_service(url: str) -> dict:
    """Controller record for an end-to-end run.

    A local controller that is down is started by ``ensure_controller``.
    A failure here is a setup error, not a model score.
    """
    return ensure_controller(url)


@solver
def require_tool_service(url: str) -> Solver:
    """Record the controller's tool list.

    A down controller or an empty tool list raises. That is not scored as a
    model failure.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set(TOOL_SERVICE_KEY, await asyncio.to_thread(prepare_tool_service, url))
        return state

    return solve


@task
def openearthagent(
    regime: Regime = "e2e",
    category: Category = "all",
    task_id: str | None = None,
    vision: bool = True,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judges: str | list[str] = "",
    controller: str = "",
    solver: Solver | Agent | None = None,
) -> Task:
    """OpenEarthAgent test split.

    Args:
        regime: "step" (teacher-forced, no tools) or "e2e" (live tools).
        category: Keep tasks that use at least one tool of this category
            (perception, operation, logic, gis), or "all".
        task_id: Restrict to one dataset ``idx``.
        vision: When false, tasks that carry an image are not applicable.
        max_iterations: End-to-end model-turn cap (upstream ``--max_rounds``, 15
            in both ``eval_e2e.sh`` and ``eval_step.sh``). Step mode records
            the same value and still generates once per gold turn. A YAML run
            config overrides it with ``task.args.max_iterations``.
        max_tokens: Per-turn generation cap (upstream OpenAI driver, 256).
        judges: Extra answer judges, comma-separated. The grader role is
            always one judge, separate from these.
        controller: Tool controller URL. Empty uses the environment or the default.
        solver: Agent to evaluate. Step mode defaults to one ``generate`` call.
            End-to-end defaults to the LangGraph ReAct agent.
    """
    if regime not in ("step", "e2e"):
        raise ValueError("regime must be 'step' or 'e2e'")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    dataset = openearthagent_dataset(
        regime,
        task_id=task_id,
        category=category,
        attach_images=vision,
    )
    revisions = _revision()
    url = controller_url(controller)
    setup: list[Solver] = [record_run_identity()]
    if not vision:
        setup.append(skip_without_vision())
    if regime == "e2e":
        setup.append(require_tool_service(url))
        setup.append(publish_mcp_servers(sample_servers_factory(url)))
        agent = solver or langgraph_react(
            max_model_calls=max_iterations, recursion_limit=4 * max_iterations + 10
        )
        scorers = [
            openearthagent_scorer(),
            *extra_answer_judges(judges),
        ]
        headline = "openearthagent_scorer.any_order"
    else:
        agent = solver or generate()
        scorers = [openearthagent_step(), *extra_answer_judges(judges, step=True)]
        headline = "openearthagent_step.tool"
    return Task(
        dataset=dataset,
        setup=setup,
        solver=agent,
        scorer=scorers,
        headline_metric=headline,
        config=GenerateConfig(temperature=DEFAULT_TEMPERATURE, max_tokens=max_tokens),
        metadata={
            "benchmark": "openearthagent",
            "summary_groups": ["task_kind"],
            "regime": regime,
            "category": category,
            "task_id": task_id,
            "vision": vision,
            "max_iterations": max_iterations,
            "max_tokens": max_tokens,
            "temperature": DEFAULT_TEMPERATURE,
            "judges": judges,
            "controller_url": url if regime == "e2e" else "",
            "openearthagent_rev": revisions["rev"],
            "dataset_revision": revisions["dataset_revision"],
            "tools_sha256": ported_code_sha256(),
            "protocol_note": (
                "Logged rates are fractions in [0, 1]. Tables 3 and 4 of "
                "arXiv:2602.17665 print percentages (times 100)."
            ),
        },
        version=1,
    )
