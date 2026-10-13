"""The A2A layer (protocol 1.0): each specialist served on localhost, one ``ask_<slug>`` tool per specialist.

Server side, ``SubAgentExecutor`` wraps one compiled LangGraph sub-agent. It runs
each request in a copy of the sample's ``contextvars.Context`` (captured by the
orchestrator), not in uvicorn's: the sample store and the open MCP sessions are
found through that context. A copy per request, so a context variable one request
sets (its own model bridge, see ``graph``) reaches neither the other requests nor
the orchestrator. ``contextId`` is the sub-agent's
``thread_id``, so a follow-up to the same specialist continues its conversation.
The reply is a completed task with one text artifact, a failed task with the
error, or, when the sub-agent answers ``NEEDS_INPUT: ...``, an
``input-required`` task whose status message says what is missing (spec 4.1.3).
The A2A ``taskId`` goes into the sub-agent run's metadata (``a2a_task_id``), so
every tool call it records names the task that caused it.

Client side, the orchestrator resolves each specialist's AgentCard over HTTP
and gets one tool per card. Its description is the card's description and
skills (name, description, first example). The tool sends ``SendMessage`` and
returns the reply text, with ``{agent, task_id, context_id, state}`` as the
ToolMessage artifact; it is tagged with the ``peer`` role, so domain traces
leave it out (``agents.trace_callback``). After an ``input-required`` reply, the next message
carries the same ``taskId`` and ``contextId``, so the task continues (spec
3.4.3). A transport error is returned as text; it never aborts the sample.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers.proto_helpers import (
    get_message_text,
    get_text_parts,
    new_message,
    new_task,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskState,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from agentic_evals.agents.trace_callback import ROLE_METADATA_KEY

HOST = "127.0.0.1"
RPC_PATH = "/"
TIMEOUT_S = 1800.0
# A sub-agent reply that starts with this ends the turn in input-required.
NEEDS_INPUT = "NEEDS_INPUT:"
CARD_VERSION = "1.0.0"
# AgentSkill fields (spec 4.4.5). Anything else in a plan's skill (``tools``) stays off the card.
_SKILL_FIELDS = ("id", "name", "description", "tags", "examples")

PEER = "peer"

# (request text, context id, task id) -> reply text
RunSubAgent = Callable[[str, str, str], Any]


class SubAgentExecutor(AgentExecutor):
    def __init__(self, run: RunSubAgent, sample_context: contextvars.Context | None = None) -> None:
        self.run = run
        self.sample_context = sample_context

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        if context.current_task is None:
            history = [context.message] if context.message is not None else []
            await event_queue.enqueue_event(new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED, history=history))
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        try:
            coro = self.run(context.get_user_input(), context_id, task_id)
            context_copy = self.sample_context.copy() if self.sample_context is not None else None
            reply = (await asyncio.create_task(coro, context=context_copy) or "").strip()
            if reply.startswith(NEEDS_INPUT):
                question = reply.removeprefix(NEEDS_INPUT).strip() or "More input is needed."
                await updater.requires_input(new_text_message(question, context_id=context_id, task_id=task_id))
                return
            await updater.add_artifact([new_text_part(reply or "(no answer)")], name="answer")
            await updater.complete()
        except Exception as exc:  # noqa: BLE001
            await updater.failed(
                new_text_message(f"{type(exc).__name__}: {exc}", context_id=context_id, task_id=task_id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def langgraph_runner(graph: Any, callbacks: list[Any], recursion_limit: int) -> RunSubAgent:
    """``RunSubAgent`` for a ``create_agent`` graph compiled with a checkpointer."""
    from agentic_evals.agents.langgraph_react import _last_ai_text

    async def run(text: str, context_id: str, task_id: str) -> str:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={
                "recursion_limit": recursion_limit,
                "callbacks": callbacks,
                "configurable": {"thread_id": context_id},
                "metadata": {"a2a_task_id": task_id},
            },
        )
        return _last_ai_text(list(result.get("messages") or []))

    return run


def agent_card(name: str, description: str, skills: list[dict[str, Any]], url: str) -> AgentCard:
    """An AgentCard with the required fields of spec 4.4.1, and skills as in 4.4.5.

    ``skills`` are plan skill dicts; only the AgentSkill fields reach the card.
    ``version`` is ``1.0.0`` plus a hash of the card text as build metadata, so a
    regenerated description shows up as a new version.
    """
    card_skills = [{key: skill[key] for key in _SKILL_FIELDS if key in skill} for skill in skills]
    text = json.dumps([name, description, card_skills], sort_keys=True)
    return AgentCard(
        name=name,
        description=description,
        version=f"{CARD_VERSION}+{hashlib.sha256(text.encode()).hexdigest()[:8]}",
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=url.rstrip("/") + RPC_PATH, protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(**skill) for skill in card_skills],
    )


@asynccontextmanager
async def serve(
    agents: dict[str, tuple[AgentExecutor, Callable[[str], AgentCard]]],
) -> AsyncIterator[dict[str, str]]:
    """Serve each ``name: (executor, card(url))`` on a free localhost port; yield ``{name: url}``."""
    running: list[tuple[uvicorn.Server, asyncio.Task]] = []
    urls: dict[str, str] = {}
    try:
        for name, (executor, card_for) in agents.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, 0))
            url = f"http://{HOST}:{sock.getsockname()[1]}"
            card = card_for(url)
            handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
            app = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, rpc_url=RPC_PATH)])
            server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            running.append((server, task))
            while not server.started:
                if task.done():
                    task.result()
                    raise RuntimeError(f"A2A server for {name!r} stopped before it started")
                await asyncio.sleep(0.02)
            urls[name] = url
        yield urls
    finally:
        for server, _ in running:
            server.should_exit = True
        await asyncio.gather(*(task for _, task in running), return_exceptions=True)


class _AskArgs(BaseModel):
    request: str = Field(
        description="What you need done, in plain words, with every file path and value the step needs."
    )


def _state(task: Any) -> str:
    # TASK_STATE_INPUT_REQUIRED -> input-required
    return TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower().replace("_", "-")


def _reply_text(event: Any) -> str:
    if event.HasField("message"):
        return get_message_text(event.message)
    task = event.task
    texts = [t for artifact in task.artifacts for t in get_text_parts(artifact.parts) if t]
    if task.status.HasField("message"):
        texts.append(get_message_text(task.status.message))
    state = _state(task)
    text = "\n".join(t for t in texts if t)
    return text if state == "completed" else f"[{state}] {text}"


def _description(card: AgentCard) -> str:
    """The card as the orchestrator's model reads it: description, then each skill and one example."""
    lines = [f"Send a request to {card.name} over A2A. {card.description}", "Skills:"]
    for skill in card.skills:
        lines.append(f"- {skill.name}: {skill.description}")
        if skill.examples:
            lines.append(f'  Example request: "{skill.examples[0]}"')
    return "\n".join(lines)


async def ask_tools(urls: dict[str, str], http: httpx.AsyncClient) -> list[BaseTool]:
    """One ``ask_<name>`` tool per A2A agent, described by the agent's card."""
    config = ClientConfig(streaming=False, httpx_client=http)
    tools: list[BaseTool] = []
    for name, url in urls.items():
        card = await A2ACardResolver(http, url).get_agent_card()
        client = await create_client(card, client_config=config)
        tools.append(_ask_tool(name, card, client))
    return tools


def _ask_tool(name: str, card: AgentCard, client: Any) -> BaseTool:
    # One A2A context per specialist for the sample; ``task`` is set only while a task waits for input.
    ids: dict[str, str | None] = {"context": None, "task": None}

    async def ask(request: str) -> tuple[str, dict[str, Any]]:
        message = new_message(
            [new_text_part(request)], context_id=ids["context"], task_id=ids["task"], role=Role.ROLE_USER
        )
        send = SendMessageRequest(message=message, configuration=SendMessageConfiguration(return_immediately=False))
        reply = ""
        sent = {"agent": name, "task_id": ids["task"], "context_id": ids["context"], "state": None}
        try:
            async for event in client.send_message(send):
                reply = _reply_text(event)
                if event.HasField("task"):
                    ids["context"] = event.task.context_id or ids["context"]
                    waiting = event.task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
                    ids["task"] = event.task.id if waiting else None
                    sent.update(task_id=event.task.id, context_id=ids["context"], state=_state(event.task))
                else:
                    ids["context"] = event.message.context_id or ids["context"]
                    ids["task"] = None
                    sent.update(context_id=ids["context"], state="message")
        except Exception as exc:  # noqa: BLE001
            return f"Error calling agent {name!r}: {type(exc).__name__}: {exc}", {**sent, "state": "error"}
        return reply or f"Agent {name!r} sent no reply.", sent

    return StructuredTool.from_function(
        coroutine=ask,
        name=f"ask_{name}"[:64],
        description=_description(card),
        args_schema=_AskArgs,
        response_format="content_and_artifact",
        metadata={ROLE_METADATA_KEY: PEER},
    )
