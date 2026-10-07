"""Optional LLM judges of the whole trajectory, reported as our additions.

These are not part of any benchmark's published protocol and are never mixed
into a benchmark's own metrics. Each judge runs as a separate scorer (one per
model), so every verdict is stored per sample and judge disagreement can be
inspected (``agentic_evals.scorers.agreement``) instead of being hidden behind a
majority vote.

The prompt and trajectory rendering are agentevals' reference-trajectory judge
(``agentevals.trajectory.llm.TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE`` and its
``_format_inputs``; langchain-ai/agentevals v0.0.9). The call goes through an
Inspect model (``model=`` or the ``grader`` role) so the judge is recorded in
the log like any other model; agentevals' own structured-output call is replaced
by asking for ``{"reasoning": ..., "score": true|false}`` JSON, which is our
choice, not agentevals'.

Tasks publish their gold trajectory as ``metadata["reference_trajectory"]``: a
list of ``{"name", "arguments", "output"}`` steps.
"""

from __future__ import annotations

import json
import math
from typing import Any

from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE, _format_inputs
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from agentic_evals.agents.trace_callback import read_trace

OUTPUT_INSTRUCTION = (
    '\nRespond with JSON only: {"reasoning": "<your reasoning>", "score": true or false}, '
    "where score is true if the trajectory is accurate."
)


def _as_openai_messages(steps: list[dict[str, Any]], final_answer: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        call_id = f"call_{i}"
        arguments = step.get("arguments")
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": step.get("name", ""),
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        output = step.get("output")
        if output is not None:
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text[:4000]})
    if final_answer:
        messages.append({"role": "assistant", "content": final_answer})
    return messages


@scorer(metrics={"trajectory_accuracy": [mean()]})
def trajectory_judge(model: str | None = None):
    """agentevals-style reference-trajectory judge.

    Args:
        model: Judge model (e.g. "openai/gpt-4.1-mini"); defaults to the ``grader`` role.
    """

    async def score(state: TaskState, target: Target) -> Score:
        reference = state.metadata.get("reference_trajectory")
        if reference is None:
            return Score(value={"trajectory_accuracy": math.nan}, explanation="no reference trajectory")
        judge = get_model(model) if model else get_model(role="grader", required=True)
        actual = [{"name": e["name"], "arguments": e["input"], "output": e.get("output")} for e in read_trace(state.store)]
        outputs, reference_outputs = _format_inputs(
            _as_openai_messages(actual, state.output.completion if state.output else ""),
            _as_openai_messages(reference, None),
        )
        prompt = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE.format(outputs=outputs, reference_outputs=reference_outputs)
        # A short JSON verdict. Leaving max_tokens unset inherits an uncapped
        # eval (`--max-tokens -1`), and the endpoint then reserves the rest of
        # the context as output, which overflows on a long trajectory.
        try:
            result = await judge.generate(
                [ChatMessageUser(content=prompt + OUTPUT_INSTRUCTION)],
                config=GenerateConfig(temperature=0, max_tokens=2048),
            )
        except Exception as exc:
            return Score(
                value={"trajectory_accuracy": math.nan},
                explanation=f"{type(exc).__name__}: {exc}"[:2000],
                metadata={"judge_model": str(judge), "judge_prompt": "agentevals TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE"},
            )
        try:
            data = json.loads(result.completion.strip().removeprefix("```json").removesuffix("```").strip())
            verdict, reasoning = float(bool(data["score"])), str(data.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            verdict, reasoning = math.nan, f"unparseable judge output: {result.completion[:500]}"
        return Score(
            value={"trajectory_accuracy": verdict},
            explanation=reasoning,
            metadata={"judge_model": str(judge), "judge_prompt": "agentevals TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE"},
        )

    return score


def extra_judges(models: str | list[str]) -> list:
    """Scorers for judge models, given as a list or a comma-separated string (empty: none).

    Inspect parses a comma-separated ``-T`` value into a list.
    """
    names = models if isinstance(models, list) else models.split(",")
    return [trajectory_judge(model=m.strip()) for m in names if m.strip()]
