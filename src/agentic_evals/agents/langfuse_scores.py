"""Inspect scores as Langfuse scores on each sample's trace.

When a sample is scored, every score its scorers produced is attached to the
sample's trace (``langfuse_trace.sample_trace_id``), so Langfuse can filter and
chart traces by outcome. Values are copied as the benchmark's scorer returned
them: one Langfuse score per scorer, or per key when a scorer returns a dict
(``<scorer>/<key>``). Nothing is combined, and a judge outside the paper's
protocol keeps its own score name.
"""

from __future__ import annotations

import os
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, hooks

from agentic_evals.agents.langfuse_trace import ENV_KEYS, sample_trace_id


def langfuse_scores(scores: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``create_score`` arguments for each scalar in an Inspect sample's scores."""
    rows: list[dict[str, Any]] = []
    for name, score in (scores or {}).items():
        value = score.value
        items = value.items() if isinstance(value, dict) else [(None, value)]
        for key, item in items:
            label = name if key is None else f"{name}/{key}"
            row: dict[str, Any] = {"name": label, "comment": score.explanation, "metadata": {"scorer": name}}
            if isinstance(item, bool):
                rows.append({**row, "value": float(item), "data_type": "BOOLEAN"})
            elif isinstance(item, (int, float)):
                rows.append({**row, "value": float(item), "data_type": "NUMERIC"})
            elif isinstance(item, str):
                rows.append({**row, "value": item, "data_type": "CATEGORICAL"})
    return rows


@hooks(name="langfuse_scores", description="Attach each sample's scores to its Langfuse trace.")
class LangfuseScores(Hooks):
    def enabled(self) -> bool:
        return all(os.environ.get(key) for key in ENV_KEYS)

    async def on_sample_end(self, data: SampleEnd) -> None:
        from langfuse import get_client

        client = get_client()
        trace_id = sample_trace_id(data.sample.uuid)
        for row in langfuse_scores(data.sample.scores):
            client.create_score(trace_id=trace_id, **row)
        client.flush()
