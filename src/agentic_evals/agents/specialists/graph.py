"""Orchestrator over tool-specialist sub-agents reached through A2A, run as an Inspect solver.

For each sample:

1. Load the task's MCP tools, as ``langgraph_react`` does.
2. Split them into specialists with the committed clusters (``clustering``);
   no embedding call is made at eval time.
3. Build one LangGraph ReAct sub-agent per specialist over its tools, and serve
   each one as an A2A server on localhost (``a2a``).
4. Run the orchestrator: a ReAct agent with no domain tools, only one
   ``ask_<specialist>`` tool per agent card. It writes the final answer.

Every model call, orchestrator's or specialist's, goes to the eval's ``--model``
through ``agent_bridge()``. Each specialist request has its own bridge
(``own_bridge``): the bridge tracks one conversation as the sample's messages and
output, and a specialist's loop would otherwise take that place from the
orchestrator. The sample's messages and final answer are the orchestrator's.
Three traces are stored per sample:

- ``TRACE_KEY`` (scored): the MCP calls only, the orchestrator's own and the
  specialists', with the caller in ``agent``. Delegations are not trajectory
  steps, so the benchmarks' metrics count the same things as for ``langgraph_react``.
- ``RAW_TRACE_KEY``: every tool call of the orchestrator, ``ask_*`` included,
  as it saw them. An ``ask_*`` entry's ``artifact`` holds the A2A task id;
  the specialist's calls for that task carry it as ``a2a_task_id``.
- ``SUBAGENT_TRACES_KEY``: per specialist, its conversation (from its
  checkpointer) and its MCP calls.

With Weave on, one ``WeaveRecorder`` serves every agent, so a specialist's run
nests under the ``ask_*`` call that reached it (``agents.weave_trace``).

The split used is stored under ``SPECIALISTS_KEY``.
"""

from __future__ import annotations

import contextvars
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from typing import Any

import httpx
from inspect_ai.agent import Agent, AgentState, agent, agent_bridge
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import store
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agentic_evals.agents.langgraph_react import (
    TERMINATION_KEY,
    _message_text,
    default_middleware,
    inspect_chat_model,
    load_task_mcp_tools,
    run_graph,
)
from agentic_evals.agents.specialists.a2a import (
    NEEDS_INPUT,
    TIMEOUT_S,
    RunSubAgent,
    SubAgentExecutor,
    agent_card,
    ask_tools,
    langgraph_runner,
    serve,
)
from agentic_evals.agents.specialists.clustering import Specialist, plan_specialists
from agentic_evals.agents.trace_callback import RAW_TRACE_KEY, SUBAGENT_TRACES_KEY, TraceRecorder
from agentic_evals.agents.weave_trace import weave_recorder

SPECIALISTS_KEY = "agentic_evals:specialists"

ORCHESTRATOR_PROMPT = """\
# How to work

Specialist agents run the domain tools. Each owns one group of tools, and you \
reach one over A2A with its `ask_<name>` tool. Each tool's description is that \
agent's card: what it does and its skills, with an example request. Break the \
task into steps and give each step to the specialist whose skills fit it. You \
also have a few tools of your own, listed after this text; use one when a step \
needs its result, for example to list a directory before you ask about a file in it.

- A specialist does not see the conversation. Put in each request everything \
the step needs: the file paths, values and earlier results it should use, and \
what it should return.
- A specialist keeps its own history, so a follow-up to the same one can refer \
to its earlier answer.
- A reply that starts with [input-required] is a question about your request: the \
specialist cannot go on without what it names. Answer it with that information, taken \
from the task or from earlier results (or get it with one of your own tools), and send \
it to the same specialist; it continues the same task. Repeating a request that was \
already answered with [input-required] gets the same reply, so each new request must \
add what was asked for.
- Check each answer before the next step. If a step failed, say what went wrong \
when you ask again, or try another specialist.
- When every step is done, give the final answer yourself, in the format the \
task asks for."""

# Added after the describer's role prompt. It is the only reply rule the protocol needs:
# the orchestrator reads the NEEDS_INPUT prefix as input-required (a2a.SubAgentExecutor).
INPUT_PROMPT = f"""\
If the request leaves out something you need, such as a file path, a band or a threshold, \
reply with one line that starts with `{NEEDS_INPUT}` and names what is missing."""

DEFAULT_ROLE_PROMPT = "You are {name}, a specialist agent."


def _question(state: AgentState) -> str:
    for message in state.messages:
        if isinstance(message, ChatMessageUser):
            return message.text
    return ""


def specialist_prompt(spec: Specialist, question: str | None) -> str:
    """The describer's role prompt, the fixed A2A rules, the tools, and the sample question."""
    role = spec.role_prompt or DEFAULT_ROLE_PROMPT.format(name=spec.name)
    prompt = f"{role}\n\n{INPUT_PROMPT}\n\nYour tools: {', '.join(spec.tools)}."
    if question:
        prompt += f"\n\nThe overall task, for context:\n\n{question}"
    return prompt


def orchestrator_prompt(own_tools: tuple[str, ...]) -> str:
    """The orchestrator's rules, plus the names of the tools it keeps for itself."""
    if not own_tools:
        return ORCHESTRATOR_PROMPT
    return f"{ORCHESTRATOR_PROMPT}\n\nYour own tools, besides the ask_ tools: {', '.join(own_tools)}."


def without_tools(specialists: list[Specialist], held: tuple[str, ...]) -> list[Specialist]:
    """The specialists with the ``held`` tools removed; one left with no tools is dropped."""
    kept = []
    for spec in specialists:
        tools = tuple(t for t in spec.tools if t not in held)
        if tools:
            kept.append(replace(spec, tools=tools))
    return kept


def own_bridge(run: RunSubAgent) -> RunSubAgent:
    """``run`` with its own ``agent_bridge()`` per request, apart from the orchestrator's.

    The executor runs each request in its own context copy, so this bridge is
    seen only by that request's model calls.
    """

    async def bridged(text: str, context_id: str, metadata: dict[str, Any]) -> str:
        async with agent_bridge():
            return await run(text, context_id, metadata)

    return bridged


def message_dict(message: Any) -> dict[str, Any]:
    """A LangChain message as role, text, tool calls and (for a tool result) the tool name."""
    turn: dict[str, Any] = {"role": message.type, "text": _message_text(message)}
    calls = [{"name": c["name"], "args": c["args"]} for c in getattr(message, "tool_calls", None) or []]
    if calls:
        turn["tool_calls"] = calls
    if message.type == "tool":
        turn["name"] = message.name
    return turn


async def subagent_traces(
    specialists: list[Specialist],
    graphs: dict[str, Any],
    prompts: dict[str, str],
    raw: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each specialist's conversation, per A2A context it served, and its MCP calls.

    The context ids come from the orchestrator's ``ask_*`` artifacts. The
    conversation is read from the specialist's checkpointer, so a run stopped by
    its recursion limit still shows the steps it took.
    """
    traces: dict[str, dict[str, Any]] = {}
    for spec in specialists:
        contexts = []
        for entry in raw:
            artifact = entry.get("artifact") or {}
            cid = artifact.get("context_id")
            if artifact.get("agent") == spec.slug and cid and cid not in contexts:
                contexts.append(cid)
        messages = []
        for cid in contexts:
            snapshot = await graphs[spec.slug].aget_state({"configurable": {"thread_id": cid}})
            messages.extend(message_dict(m) for m in snapshot.values.get("messages", []))
        traces[spec.slug] = {
            "name": spec.name,
            "tools": list(spec.tools),
            "system_prompt": prompts[spec.slug],
            "context_ids": contexts,
            "messages": messages,
            "calls": [e for e in entries if e["agent"] == spec.slug],
        }
    return traces


@agent
def langgraph_specialists(
    clusters: str = "auto",
    recompute: bool = False,
    share_question: bool = False,
    recursion_limit: int = 50,
    sub_recursion_limit: int = 30,
    max_model_calls: int | None = None,
    orchestrator_tools: tuple[str, ...] = ("get_filelist",),
    name: str = "orchestrator",
) -> Agent:
    """Orchestrator plus one A2A specialist per cluster of the task's tools.

    Args:
        clusters: ``"auto"`` uses the committed file in
            ``agents/specialists/clusters/`` that covers the sample's tools; or a
            path to one such file.
        recompute: When no file covers the tools, cluster and describe them
            (the endpoints in ``embedding.yaml``) and write the file, instead
            of failing.
        share_question: Also put the sample's question in each specialist's system
            prompt. Off by default: a specialist sees only the orchestrator's request,
            so it answers the step and not the task's answer format.
        recursion_limit: Orchestrator's LangGraph super-step limit.
        sub_recursion_limit: Each specialist's super-step limit per request.
        max_model_calls: Orchestrator model-turn cap, as for ``langgraph_react``.
        orchestrator_tools: Task tools the orchestrator runs itself, not a
            specialist. They are taken out of the specialists' sets.
        name: Orchestrator name.
    """

    async def execute(state: AgentState) -> AgentState:
        # Domain calls of every agent share ``entries``; ``raw`` is the orchestrator's view.
        entries: list[dict[str, Any]] = []
        recorder = TraceRecorder(agent_name=name, entries=entries)
        raw = TraceRecorder(agent_name=name, roles=None)
        weave = [r for r in (weave_recorder(),) if r is not None]
        async with agent_bridge(state) as bridge:
            async with AsyncExitStack() as stack:
                tools = await load_task_mcp_tools(stack)
                by_name = {t.name: t for t in tools}
                missing = set(orchestrator_tools) - by_name.keys()
                if missing:
                    raise ValueError(f"orchestrator_tools not in the task's tools: {sorted(missing)}")
                plan = await plan_specialists(tools, clusters, recompute)
                specialists = without_tools(plan, orchestrator_tools)
                store().set(SPECIALISTS_KEY, [asdict(s) for s in specialists])

                question = _question(state) if share_question else None
                sample_context = contextvars.copy_context()
                agents, graphs, prompts = {}, {}, {}
                for spec in specialists:
                    owned = [by_name[t] for t in spec.tools]
                    prompts[spec.slug] = specialist_prompt(spec, question)
                    graphs[spec.slug] = graph = create_agent(
                        inspect_chat_model(),
                        tools=owned,
                        system_prompt=prompts[spec.slug],
                        middleware=default_middleware(None),
                        checkpointer=InMemorySaver(),
                        name=spec.slug,
                    )
                    run = own_bridge(
                        langgraph_runner(
                            graph, [TraceRecorder(agent_name=spec.slug, entries=entries), *weave], sub_recursion_limit
                        )
                    )
                    agents[spec.slug] = (
                        SubAgentExecutor(run, sample_context),
                        lambda url, spec=spec: agent_card(spec.name, spec.description, list(spec.skills), url),
                    )
                urls = await stack.enter_async_context(serve(agents))
                http = await stack.enter_async_context(httpx.AsyncClient(timeout=TIMEOUT_S))
                orchestrator = create_agent(
                    inspect_chat_model(),
                    tools=[*await ask_tools(urls, http), *(by_name[t] for t in orchestrator_tools)],
                    middleware=default_middleware(max_model_calls),
                    name=name,
                )
                termination = await run_graph(
                    orchestrator,
                    bridge,
                    state,
                    recorder,
                    recursion_limit,
                    system_text=orchestrator_prompt(orchestrator_tools),
                    callbacks=[recorder, raw, *weave],
                )
                subagents = await subagent_traces(specialists, graphs, prompts, raw.entries, entries)
            recorder.save()
            raw.save(RAW_TRACE_KEY)
            store().set(SUBAGENT_TRACES_KEY, subagents)
            store().set(TERMINATION_KEY, termination)
            return bridge.state

    return execute
