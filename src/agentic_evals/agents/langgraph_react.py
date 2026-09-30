"""Single-agent LangGraph ReAct baseline, run as an Inspect solver.

This mirrors the upstream Earth-Agent scaffold (``scripts/langchain_<model>*.py``:
a LangGraph ReAct agent over the task's MCP tools, ``recursion_limit=50``) in
langchain 1.x: ``langchain.agents.create_agent`` replaces the pre-1.0
``langgraph.prebuilt.create_react_agent``. It follows Inspect's LangChain bridge
example (``examples/bridge/langchain``): ``ChatOpenAI(model="inspect")`` inside
``agent_bridge()`` is routed to whatever ``--model`` the eval uses.

The agent knows nothing about benchmarks. The prompt comes from the sample input
and the tools from the MCP servers the task published (``agentic_evals.tooling``).
Generation settings (e.g. ``--temperature 0.1``, upstream's value) are set on the
Inspect command line; the bridge does not forward client-side settings. A call
written only as text is left as text, except Mistral's ``[TOOL_CALLS]name[ARGS]{...}``
envelope, which is turned into a structured tool call. Upstream's
``create_react_agent`` ends the turn unless the provider returns a structured tool call.
"""

from __future__ import annotations

import json
import re
from contextlib import AsyncExitStack
from typing import Any
from uuid import uuid4

from inspect_ai.agent import Agent, AgentState, agent, agent_bridge
from inspect_ai.model import ChatMessageAssistant, ModelOutput, messages_to_openai
from inspect_ai.util import store
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, convert_to_messages
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

from agentic_evals.agents.trace_callback import TraceRecorder
from agentic_evals.tooling import mcp_servers

TERMINATION_KEY = "agentic_evals:termination"

# LangChain ModelCallLimitMiddleware (exit_behavior="end") injects this prefix.
MODEL_CALL_LIMIT_PREFIX = "Model call limits exceeded"
# OpenAIFunctionCallEnv writes this when max_iterations is hit after tools ran.
REACHED_MAXIMUM_ITERATIONS = "Reached maximum iterations."


def truncated_final_answer(last_message: str, *, tool_calls: int) -> str | None:
    """Released truncated answer when the model-turn cap stops a run that used tools.

    ``OpenAIFunctionCallEnv.step`` (``exhausted`` and ``steps`` both set) writes
    ``Reached maximum iterations.`` A run that never called a tool is left unchanged.
    """
    if tool_calls > 0 and last_message.startswith(MODEL_CALL_LIMIT_PREFIX):
        return REACHED_MAXIMUM_ITERATIONS
    return None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _message_text(message)
    return ""


def _skip_output_schema_validation(session: ClientSession) -> None:
    # mcp >= 1.13 validates structured tool output against the declared output
    # schema client-side and raises, which aborts the sample. Upstream tools
    # (e.g. Earth-Agent returning NaN as null in a list[float]) were run with an
    # older client that passed the raw result to the model; keep that behaviour.
    for name in list(session._tool_output_schemas):
        session._tool_output_schemas[name] = None


# vLLM's Mistral parser: the model emits ``[TOOL_CALLS]name[ARGS]{...}``, not an
# OpenAI tool_calls array (vllm/tool_parsers/mistral_tool_parser.py).
# Venice and DeepInfra reject a name outside ``^[A-Za-z0-9_-]{1,64}$``.
_MISTRAL_TOOL_MARKER = "[TOOL_CALLS]"
_MISTRAL_ARGS_MARKER = "[ARGS]"
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tool_call(name: str, args: dict[str, Any], index: int, call_id: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": call_id or f"call_mistral_{index}",
        "type": "tool_call",
    }


def mistral_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Structured calls from Mistral's textual tool envelope. Empty when it is not one.

    Two shapes are accepted. Current checkpoints use ``[TOOL_CALLS]name[ARGS]{json}``
    (vLLM ``mistral_tool_parser``). Older ones put a JSON array of
    ``{"name", "arguments"}`` after the same marker.
    """
    if _MISTRAL_TOOL_MARKER not in text:
        return []
    calls: list[dict[str, Any]] = []
    rest = text
    while _MISTRAL_TOOL_MARKER in rest:
        _, _, rest = rest.partition(_MISTRAL_TOOL_MARKER)
        rest = rest.lstrip()
        if rest.startswith("["):
            try:
                parsed, end = json.JSONDecoder().raw_decode(rest)
            except json.JSONDecodeError:
                break
            if not isinstance(parsed, list):
                break
            for item in parsed:
                call = _call_from_mistral_object(item, len(calls))
                if call is not None:
                    calls.append(call)
            rest = rest[end:]
            continue
        if _MISTRAL_ARGS_MARKER not in rest:
            break
        name, _, after = rest.partition(_MISTRAL_ARGS_MARKER)
        name = name.strip()
        after = after.lstrip()
        try:
            args, end = json.JSONDecoder().raw_decode(after)
        except json.JSONDecodeError:
            break
        if _TOOL_NAME.fullmatch(name) and isinstance(args, dict):
            calls.append(_tool_call(name, args, len(calls)))
        rest = after[end:]
    return calls


def _call_from_mistral_object(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    args = item.get("arguments", item.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not _TOOL_NAME.fullmatch(name) or not isinstance(args, dict):
        return None
    return _tool_call(name, args, index)


def _args_dict(args: Any) -> dict[str, Any] | None:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_mistral_message(message: AIMessage) -> AIMessage:
    """Strip illegal whitespace from tool names, and parse a bare Mistral envelope.

    A structured call whose name is ``reject_task`` plus a newline is what Venice
    and DeepInfra answer with HTTP 400. When the envelope arrives as assistant
    text and ``tool_calls`` is empty, the agent would otherwise end the turn.
    """
    calls: list[dict[str, Any]] = []
    changed = False
    for call in message.tool_calls or []:
        name = str(call.get("name") or "")
        stripped = name.strip()
        if stripped != name and _TOOL_NAME.fullmatch(stripped):
            call = {**call, "name": stripped}
            changed = True
        calls.append(call)

    invalid = list(message.invalid_tool_calls or [])
    kept_invalid: list[dict[str, Any]] = []
    for call in invalid:
        name = str(call.get("name") or "").strip()
        args = _args_dict(call.get("args"))
        already = any(existing.get("id") == call.get("id") for existing in calls)
        if _TOOL_NAME.fullmatch(name) and args is not None and not already:
            calls.append(_tool_call(name, args, len(calls), call_id=call.get("id") or None))
            changed = True
            continue
        kept_invalid.append(call)

    content = message.content if isinstance(message.content, str) else message.content
    if not calls and isinstance(content, str):
        parsed = mistral_tool_calls_from_text(content)
        if parsed:
            marker = content.find(_MISTRAL_TOOL_MARKER)
            content = content[:marker].strip()
            calls = parsed
            changed = True

    if not changed:
        return message
    return message.model_copy(
        update={
            "content": content,
            "tool_calls": calls,
            "invalid_tool_calls": kept_invalid,
        }
    )


class NormalizeMistralToolCalls(AgentMiddleware):
    """Rewrite Mistral tool calls before LangGraph looks the tool up by name."""

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        return _normalize_model_response(response)


def _normalize_model_response(response: Any) -> Any:
    if isinstance(response, AIMessage):
        return normalize_mistral_message(response)
    if isinstance(response, ModelResponse):
        return ModelResponse(
            result=[_normalize_message(message) for message in response.result],
            structured_response=response.structured_response,
        )
    model_response = getattr(response, "model_response", None)
    if isinstance(model_response, ModelResponse):
        response.model_response = _normalize_model_response(model_response)
    return response


def _normalize_message(message: BaseMessage) -> BaseMessage:
    if isinstance(message, AIMessage):
        return normalize_mistral_message(message)
    return message


class ToolImagesAsUserMessage(AgentMiddleware):
    """Move images a tool returns into a user message right after the tool result.

    Chat-completions APIs accept images in user messages but not in tool messages.
    """

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if not isinstance(result, ToolMessage) or not isinstance(result.content, list):
            return result
        images = [b for b in result.content if isinstance(b, dict) and b.get("type") == "image"]
        if not images:
            return result
        text = [b for b in result.content if not (isinstance(b, dict) and b.get("type") == "image")]
        tool_message = result.model_copy(update={"content": text})
        return Command(update={"messages": [tool_message, HumanMessage(content=images)]})


async def _tool_errors_to_model(request: MCPToolCallRequest, handler):
    """Return tool exceptions to the model as error results instead of aborting the sample."""
    try:
        return await handler(request)
    except Exception as exc:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error calling tool '{request.name}': {type(exc).__name__}: {exc}")],
            isError=True,
        )


@agent
def langgraph_react(
    recursion_limit: int = 50,
    max_model_calls: int | None = None,
    name: str = "langgraph_react",
) -> Agent:
    """LangGraph ReAct agent over the task's MCP tools.

    Args:
        recursion_limit: LangGraph super-step limit (upstream Earth-Agent uses 50).
            When ``max_model_calls`` is set, this must stay above that cap so the
            middleware, not a graph recursion error, is what stops the run.
        max_model_calls: Stop after this many model turns (LangChain
            ``ModelCallLimitMiddleware`` ``run_limit``, ``exit_behavior="end"``).
            If tools ran, the injected limit message is replaced with
            ``Reached maximum iterations.``
        name: Agent name recorded on every trace entry.
    """

    async def execute(state: AgentState) -> AgentState:
        recorder = TraceRecorder(agent_name=name)
        async with agent_bridge(state) as bridge:
            servers = mcp_servers()
            async with AsyncExitStack() as stack:
                client = MultiServerMCPClient(servers)
                tools = []
                for server_name in servers:
                    session = await stack.enter_async_context(client.session(server_name))
                    tools.extend(
                        await load_mcp_tools(
                            session, server_name=server_name, tool_interceptors=[_tool_errors_to_model]
                        )
                    )
                    _skip_output_schema_validation(session)

                middleware: list[AgentMiddleware] = [
                    NormalizeMistralToolCalls(),
                    ToolImagesAsUserMessage(),
                ]
                if max_model_calls is not None:
                    middleware.append(
                        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")
                    )
                graph = create_agent(
                    ChatOpenAI(model="inspect", api_key="inspect"),
                    tools=tools,
                    middleware=middleware,
                    name=name,
                )
                messages = convert_to_messages(await messages_to_openai(state.messages))
                termination = "completed"
                try:
                    result = await graph.ainvoke(
                        {"messages": messages},
                        config={
                            "recursion_limit": recursion_limit,
                            "callbacks": [recorder],
                            "configurable": {"thread_id": str(uuid4())},
                        },
                    )
                except GraphRecursionError:
                    termination = "recursion_limit"
                    result = None
                replacement = None
                if result is not None:
                    replacement = truncated_final_answer(
                        _last_ai_text(list(result.get("messages") or [])),
                        tool_calls=len(recorder.entries),
                    )
                if replacement is not None:
                    model_name = bridge.state.output.model or ""
                    bridge.state.messages.append(
                        ChatMessageAssistant(content=replacement, model=model_name, source="generate")
                    )
                    bridge.state.output = ModelOutput.from_content(model=model_name, content=replacement)
                    termination = "max_model_calls"
            recorder.save()
            store().set(TERMINATION_KEY, termination)
            return bridge.state

    return execute
