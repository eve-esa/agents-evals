"""Weave calls for LangGraph agents: one per agent run, model turn and tool call.

inspect-wandb's Weave hooks log each sample as an Evaluation ``predict`` call,
and its OpenAI patch logs every model request under the solver's call, flat.
With several agents in a sample, that tree does not show who did what. This
LangChain callback adds the structure:

    <solver call>
    └─ orchestrator              agent run
       ├─ orchestrator · model   one model turn; the patched OpenAI calls nest here
       ├─ get_filelist           tool (an MCP tool, role "domain")
       └─ ask_raster_index       tool (A2A, role "peer")
          └─ raster_index        the specialist's run for that request
             ├─ raster_index · model
             └─ compute_tvdi

A sub-agent's run nests under the tool call that reached it when its run
metadata names that call's LangChain run id (``CALLER_RUN_ID``, sent over A2A).
One recorder serves every agent of a sample, so it can resolve that id. It is
created only when Weave is initialised (``weave_recorder``).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from weave.trace.context import call_context, weave_client_context

from agentic_evals.agents.trace_callback import DOMAIN, _role, _text

# Run metadata key: the LangChain run id of the tool call that started this run.
CALLER_RUN_ID = "caller_run_id"


def _turn(message: Any) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": getattr(message, "type", "message"), "content": _text(message)[0]}
    calls = [{"name": c["name"], "args": c["args"]} for c in getattr(message, "tool_calls", None) or []]
    if calls:
        turn["tool_calls"] = calls
    return turn


class WeaveRecorder(AsyncCallbackHandler):
    # Inline, in the caller's task: a model call pushed on Weave's call stack in
    # on_chat_model_start is the parent of the OpenAI calls the client makes next.
    run_inline = True

    def __init__(self, client: Any, parent: Any = None) -> None:
        """``parent`` is the call agent runs nest under unless reached over A2A."""
        self.client = client
        self.parent = parent
        self._calls: dict[str, Any] = {}  # run id -> open Weave call
        self._agent: dict[str, tuple[str, Any]] = {}  # run id -> (agent name, agent call)

    def _start(self, op: str, inputs: dict[str, Any], parent: Any, name: str, **attributes: Any) -> Any:
        return self.client.create_call(
            op, inputs=inputs, parent=parent or self.parent, display_name=name, attributes=attributes, use_stack=False
        )

    def _finish(self, run_id: UUID, output: Any = None, error: BaseException | None = None) -> Any:
        call = self._calls.pop(str(run_id), None)
        if call is not None:
            self.client.finish_call(call, output=output, exception=error)
        return call

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:  # a graph node: it belongs to its root's agent
            owner = self._agent.get(str(parent_run_id))
            if owner is not None:
                self._agent[str(run_id)] = owner
            return
        name = kwargs.get("name") or (serialized or {}).get("name") or "agent"
        caller = (metadata or {}).get(CALLER_RUN_ID)
        messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
        call = self._start(
            f"agent.{name}", {"messages": [_turn(m) for m in messages]}, self._calls.get(caller or ""), name, agent=name
        )
        self._calls[str(run_id)] = call
        self._agent[str(run_id)] = (name, call)

    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if str(run_id) not in self._calls:
            return
        messages = outputs.get("messages", []) if isinstance(outputs, dict) else []
        self._finish(run_id, {"answer": _text(messages[-1])[0] if messages else None})

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, error=error)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        agent, parent = self._agent.get(str(parent_run_id), ("agent", None))
        call = self._start("model", {"messages": [_turn(m) for m in messages[0]]}, parent, f"{agent} · model", agent=agent)
        self._calls[str(run_id)] = call
        call_context.push_call(call)

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(generation, "message", None)
        output = _turn(message) if message is not None else {"content": getattr(generation, "text", None)}
        if getattr(message, "usage_metadata", None):
            output["usage"] = message.usage_metadata
        call = self._finish(run_id, output)
        if call is not None:
            call_context.pop_call(call.id)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._finish(run_id, error=error)
        if call is not None:
            call_context.pop_call(call.id)

    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        agent, parent = self._agent.get(str(parent_run_id), ("agent", None))
        name = (serialized or {}).get("name") or kwargs.get("name", "tool")
        role = _role(metadata)
        inputs = inputs if inputs is not None else {"input": input_str}
        self._calls[str(run_id)] = self._start(f"tool.{name}", inputs, parent, name, agent=agent, role=role)

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        text, error = _text(output)
        result: dict[str, Any] = {"output": text, "error": error}
        artifact = getattr(output, "artifact", None)
        call = self._calls.get(str(run_id))
        if artifact is not None and call is not None and call.attributes.get("role") != DOMAIN:
            result["a2a"] = artifact
        self._finish(run_id, result)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, error=error)


def weave_recorder() -> WeaveRecorder | None:
    """A recorder under the current Weave call, or ``None`` when Weave is not initialised."""
    client = weave_client_context.get_weave_client()
    if client is None:
        return None
    return WeaveRecorder(client, call_context.get_current_call())
