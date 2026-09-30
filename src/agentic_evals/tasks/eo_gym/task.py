"""EO-Gym (arXiv:2605.01250) as an Inspect task.

Requires EO-Gym's tool server running with its data (``scripts/start_tool_server.sh``
in ``third_party/EO-Gym``) and a vision-capable model under test::

    uv run inspect eval agentic_evals/eo_gym --model openai-api/<name>/<vlm> \\
        -M strict_tools=false --model-role grader=openai/gpt-4.1-mini

Each epoch is one independent trajectory (EO-Gym's traj0..traj{n-1}); defaults
follow the released evaluation: 3 trajectories, temperature 0.3, at most 15
model turns (``OpenAIFunctionCallEnv.max_iterations``), the "simple" prompt
and the dataset-mapped ("skill") tools. A turn that still calls tools after
that budget stops with ``Reached maximum iterations.``
Only the Verified tool mode exists in the released code (perception tools are
hard-wired to ground-truth-backed outputs), so ``tool_mode="unverified"`` is
rejected rather than silently run as verified.

EO-Gym puts imagery in the prompt, so a text-only model cannot read the tasks:
with ``vision=false`` no agent is run and every task is recorded as not
applicable (unscored), which keeps such models visible in the results.
"""

from __future__ import annotations

import tomllib

from inspect_ai import Epochs, Task, task
from inspect_ai.agent import Agent
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agentic_evals.agents.langgraph_react import langgraph_react
from agentic_evals.paths import REPO_ROOT
from agentic_evals.run_identity import record_run_identity
from agentic_evals.tasks.eo_gym.dataset import PromptVariant, Toolset, eo_gym_dataset
from agentic_evals.scorers.judges import extra_judges
from agentic_evals.tasks.eo_gym.scorers import NOT_APPLICABLE_KEY, eo_gym_early_stop, eo_gym_scorer
from agentic_evals.tasks.eo_gym.tools import check_tool_server, sample_servers_factory
from agentic_evals.tooling import publish_mcp_servers

DEFAULT_EXECUTE_URL = "http://127.0.0.1:8081/execute"


@solver
def mark_not_applicable(reason: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set(NOT_APPLICABLE_KEY, reason)
        state.completed = True
        return state

    return solve


@solver
def require_tool_server(execute_url: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        check_tool_server(execute_url)
        return state

    return solve


@task
def eo_gym(
    prompt: PromptVariant = "simple",
    toolset: Toolset = "skill",
    tool_mode: str = "verified",
    trajectories: int = 3,
    max_calls: int = 15,
    eo_task: str | None = None,
    vision: bool = True,
    execute_url: str = DEFAULT_EXECUTE_URL,
    trajectory_judges: str | list[str] = "",
    solver: Solver | Agent | None = None,
) -> Task:
    """EO-Gym test split (1,436 tasks) with EO-Gym's tools over MCP.

    Args:
        prompt: "simple" (main setting) or "complex" (the paper's Detailed prompt).
        toolset: "skill" (dataset-mapped tools, main setting) or "all" (35 tools).
        tool_mode: Only "verified" is available in the released code.
        trajectories: Trajectories per task (epochs); Pass@k up to this k.
        max_calls: Model-turn budget per trajectory (released ``max_iterations``, default 15).
        eo_task: Restrict to one EO task family (e.g. "Temporal Reasoning").
        vision: Whether the model under test accepts images; if false, tasks are recorded as not applicable.
        execute_url: EO-Gym tool server endpoint.
        trajectory_judges: Comma-separated judge models for optional whole-trajectory LLM judges
            (our addition, one scorer per judge; see agentic_evals.scorers.judges).
        solver: Agent to evaluate (default: LangGraph ReAct).
    """
    if tool_mode != "verified":
        raise ValueError(
            "EO-Gym's released tools only implement the Verified mode (OPTICAL_BBOX_BACKEND is hard-coded "
            "to 'ground_truth'); an Unverified run would silently be Verified."
        )
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    if vision:
        setup = [record_run_identity(), require_tool_server(execute_url), publish_mcp_servers(sample_servers_factory(execute_url))]
        # The middleware stops after max_calls model turns. The recursion limit
        # only has to stay above that (model node + tool node per turn).
        agent = solver or langgraph_react(max_model_calls=max_calls, recursion_limit=4 * max_calls + 10)
    else:
        setup = [record_run_identity()]
        agent = mark_not_applicable("EO-Gym requires a vision-capable model (imagery is in the prompt)")
    return Task(
        dataset=eo_gym_dataset(prompt, toolset, eo_task, include_images=vision),
        setup=setup,
        solver=agent,
        scorer=[eo_gym_scorer(), *extra_judges(trajectory_judges)],
        epochs=Epochs(trajectories, eo_gym_early_stop(max_traj=trajectories)),
        config=GenerateConfig(temperature=0.3, max_tokens=2048),
        metadata={
            "benchmark": "eo_gym",
            "prompt": prompt,
            "toolset": toolset,
            "tool_mode": tool_mode,
            "trajectories": trajectories,
            "max_calls": max_calls,
            "vision": vision,
            "trajectory_judges": trajectory_judges,
            "eo_gym_revision": lock["eo_gym"]["revision"],
        },
        version=1,
    )
