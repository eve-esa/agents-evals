"""Agent trace view for the workshop notebook.

The renderer ``visualize_agent_trace`` is taken from the EVE notebook
``Agentic_EVE.ipynb`` in https://github.com/eve-esa/phinnovation (cell "Tracing").
Changes from that notebook: the timeline bar is drawn only when the caller
gives latencies (Inspect does not time tool calls), and the adapter
``trace_from_sample`` turns an Inspect sample into the shape it reads.

For a run of the A2A specialists solver (``langgraph_specialists``), each
``ask_*`` call is drawn as one A2A delegation with the specialist's own model
and tool steps nested inside it, and a timeline gives each agent its own row.
"""

import ast as _ast
import html
import json
import re
from typing import Any, Dict, List, Tuple

from IPython.display import HTML, display


def _escape(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


# ── JSON syntax highlighting ─────────────────────────────────────────────────
_JSON_TOKEN_RE = re.compile(
    r'("(?:[^"\\]|\\.)*")(\s*:)'   # object key  → group 1 + 2
    r'|("(?:[^"\\]|\\.)*")'         # string value → group 3
    r'|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'  # number → group 4
    r'|\b(true|false|null)\b'        # keyword     → group 5
)


def _highlight_json(s: str) -> str:
    """Return HTML with inline-style JSON syntax colouring (no <style> block)."""
    out, last = [], 0
    for m in _JSON_TOKEN_RE.finditer(s):
        out.append(html.escape(s[last:m.start()]))
        last = m.end()
        g1, g2, g3, g4, g5 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if g1:   # key
            out.append(
                f'<span style="color:#6d28d9;font-weight:600">{html.escape(g1)}</span>'
                f'<span style="color:#475569">{html.escape(g2)}</span>'
            )
        elif g3: # string value
            out.append(f'<span style="color:#15803d">{html.escape(g3)}</span>')
        elif g4: # number
            out.append(f'<span style="color:#c2410c">{html.escape(g4)}</span>')
        elif g5: # true / false / null
            out.append(f'<span style="color:#0e7490;font-weight:600">{html.escape(g5)}</span>')
    out.append(html.escape(s[last:]))
    return "".join(out)


def _extract_tool_output(raw: Any) -> tuple:
    """
    Unwrap a tool output value into (display_text, is_json).

    Handles:
    - Plain strings (JSON or arbitrary text)
    - Python-repr strings of lists/dicts (single-quoted, from str() / repr())
    - LangChain content blocks: [{'type': 'text', 'text': '...', 'id': '...'}]
    - Already-parsed dicts / lists
    """
    if raw is None:
        return "", False

    # Coerce string reprs to Python objects
    if isinstance(raw, str):
        stripped = raw.strip()
        # Try Python literal eval first (handles single-quoted reprs)
        try:
            raw = _ast.literal_eval(stripped)
        except Exception:
            # Fall back to JSON
            try:
                raw = json.loads(stripped)
            except Exception:
                # Plain text: pretty-print if it looks like JSON
                if stripped and stripped[0] in ('{', '['):
                    try:
                        return json.dumps(json.loads(stripped), indent=2, ensure_ascii=False), True
                    except Exception:
                        pass
                return raw, False

    # Unwrap LangChain content blocks: [{'type': 'text', 'text': '...'}]
    if (
        isinstance(raw, list)
        and raw
        and all(isinstance(x, dict) and x.get("type") == "text" for x in raw)
    ):
        texts = [x.get("text", "") for x in raw]
        joined = "\n".join(texts)
        try:
            return json.dumps(json.loads(joined), indent=2, ensure_ascii=False), True
        except Exception:
            return joined, False

    # Regular dict / list
    try:
        return json.dumps(raw, indent=2, ensure_ascii=False, default=str), True
    except Exception:
        return str(raw), False


def _pair_tool_inputs(trace: List[Dict[str, Any]]) -> Dict[int, Any]:
    """Map each tools-step index to the args dict from the preceding agent tool_calls queue."""
    queue: List[Tuple[Any, Any]] = []
    out: Dict[int, Any] = {}
    for i, step in enumerate(trace):
        node = (step.get("node") or "").lower()
        if node == "agent":
            for tc in step.get("tool_calls") or []:
                queue.append((tc.get("name"), tc.get("args")))
        elif node in ("tools", "a2a"):
            name = step.get("name")
            args = None
            for j, (qn, qa) in enumerate(queue):
                if qn == name:
                    args = qa
                    queue.pop(j)
                    break
            if args is None and queue:
                args = queue.pop(0)[1]
            out[i] = args
    return out


# ── Inline-style constants (no <style> block - GitHub-safe) ──────────────────
_STEP_STYLES = {
    "agent": {
        "bg": "#f5f3ff",
        "rail": "#7c3aed",
        "pill_bg": "#ede9fe",
        "pill_fg": "#5b21b6",
        "title": "EVE Agent Step",
        "sub": "Plans the next action or writes the final reply.",
        "gantt": "#7c3aed",
    },
    "tools": {
        "bg": "#ecfeff",
        "rail": "#0891b2",
        "pill_bg": "#cffafe",
        "pill_fg": "#155e75",
        "title": "Tool Execution",
        "sub": "Runs one tool and returns its payload.",
        "gantt": "#0891b2",
    },
    "a2a": {
        "bg": "#fffbeb",
        "rail": "#d97706",
        "pill_bg": "#fef3c7",
        "pill_fg": "#92400e",
        "title": "A2A Delegation",
        "sub": "Sends one task to a specialist agent over A2A and waits for its reply.",
        "gantt": "#d97706",
    },
}
_DEFAULT_STYLE = {
    "bg": "#f8fafc", "rail": "#64748b",
    "pill_bg": "#f1f5f9", "pill_fg": "#475569",
    "title": "Step", "sub": "", "gantt": "#94a3b8",
}
# A2A task state -> pill colour
_STATE_COLOURS = {
    "completed": "#15803d",
    "input-required": "#b45309",
    "failed": "#b91c1c",
    "error": "#b91c1c",
}

# Reusable inline style strings
_PRE  = "margin:0;padding:0.45rem 0.65rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:0 6px 6px 0;font-size:0.7rem;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#334155;font-family:ui-monospace,'SFMono-Regular',Consolas,monospace"
_LBL  = "font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#64748b;display:block;margin:0.55rem 0 0.2rem"
_CODE = "background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:0.78rem;color:#1e293b;font-family:ui-monospace,'SFMono-Regular',Consolas,monospace"


def _tool_body(step: Dict[str, Any], inp: Any) -> List[str]:
    tname = _escape(step.get("name", "tool"))
    parts = [
        f'<p style="margin:0 0 0.3rem;font-size:0.84rem;color:#1e293b">Tool: '
        f'<code style="background:#cffafe;color:#155e75;padding:2px 8px;'
        f'border-radius:5px;font-size:0.8rem;font-family:ui-monospace,monospace">{tname}</code></p>'
    ]
    if inp is not None:
        try:
            inp_txt = json.dumps(inp, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            inp_txt = str(inp)
        parts.append(
            f'<span style="{_LBL}">Input - arguments passed to the tool</span>'
            f'<pre style="{_PRE};border-left:3px solid #7c3aed">{_escape(inp_txt)}</pre>'
        )
    out_txt, is_json = _extract_tool_output(step.get("content"))
    out_body = _highlight_json(out_txt) if is_json else _escape(out_txt)
    parts.append(
        f'<span style="{_LBL}">Output - value returned to the model</span>'
        f'<pre style="{_PRE};border-left:3px solid #0891b2">{out_body}</pre>'
    )
    return parts


def _a2a_body(step: Dict[str, Any], number: str) -> List[str]:
    """The request the orchestrator sent, the specialist's own steps, and the reply."""
    task = step.get("a2a") or {}
    state = str(task.get("state") or "?")
    colour = _STATE_COLOURS.get(state, "#475569")
    ids = " · ".join(
        f'{label} <code style="{_CODE}">{_escape(str(task[key])[:8])}…</code>'
        for label, key in (("task", "task_id"), ("context", "context_id"))
        if task.get(key)
    )
    parts = [
        f'<p style="margin:0 0 0.3rem;font-size:0.84rem;color:#1e293b">To: '
        f'<strong>{_escape(step.get("agent_name") or step.get("agent"))}</strong> via '
        f'<code style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:5px;'
        f'font-size:0.8rem;font-family:ui-monospace,monospace">{_escape(step.get("name"))}</code> '
        f'<span style="margin-left:0.3rem;font-size:0.7rem;font-weight:700;padding:0.1rem 0.45rem;'
        f'border-radius:999px;color:#fff;background:{colour}">{_escape(state)}</span></p>',
        f'<p style="margin:0;font-size:0.72rem;color:#64748b">{ids}</p>' if ids else "",
        f'<span style="{_LBL}">Request - the A2A message the orchestrator wrote</span>'
        f'<pre style="{_PRE};border-left:3px solid #d97706">{_escape(step.get("request") or "")}</pre>',
    ]
    sub = step.get("subtrace") or []
    if sub:
        n_model = sum(1 for s in sub if s.get("node") == "agent")
        n_tools = sum(1 for s in sub if s.get("node") == "tools")
        parts.append(
            '<details open style="margin-top:0.6rem"><summary style="cursor:pointer;font-size:0.78rem;'
            f'color:#92400e;font-weight:600">Inside the specialist: {n_model} model call(s), '
            f'{n_tools} MCP tool call(s)</summary>'
            '<div style="margin:0.5rem 0 0 0.3rem;padding-left:0.7rem;border-left:2px dashed #f59e0b">'
            + "".join(_cards(sub, prefix=f"{number}."))
            + "</div></details>"
        )
    else:
        parts.append(
            '<p style="margin:0.5rem 0 0;font-size:0.75rem;color:#94a3b8">'
            "No specialist steps recorded for this task.</p>"
        )
    out_txt, _ = _extract_tool_output(step.get("content"))
    parts.append(
        f'<span style="{_LBL}">Reply - the task\'s artifact, returned to the orchestrator as the tool result</span>'
        f'<pre style="{_PRE};border-left:3px solid #0891b2">{_escape(out_txt)}</pre>'
    )
    return parts


def _cards(trace: List[Dict[str, Any]], prefix: str = "") -> List[str]:
    """One card per step. Nested steps (``prefix`` set) are numbered ``1.1``, ``1.2`` and have no elapsed footer."""
    tool_inputs = _pair_tool_inputs(trace)
    cards_html, cumulative = [], 0.0
    for i, step in enumerate(trace):
        node = (step.get("node") or "step").lower()
        s    = _STEP_STYLES.get(node, _DEFAULT_STYLE)
        lat  = float(step.get("latency_s") or 0)
        cumulative += lat
        role = _escape(step.get("role", ""))
        number = f"{prefix}{i + 1}"

        body_parts: list[str] = []

        if node == "agent":
            tcalls = step.get("tool_calls") or []
            if tcalls:
                names = ", ".join(f'<code style="{_CODE}">{_escape(tc.get("name","?"))}</code>' for tc in tcalls)
                body_parts.append(
                    f'<p style="margin:0.25rem 0;font-size:0.81rem;color:#475569">'
                    f'Requesting tool(s): {names}</p>'
                )
            content = (step.get("content") or "").strip()
            if content:
                body_parts.append(
                    f'<details style="margin-top:0.4rem"><summary style="cursor:pointer;'
                    f'font-size:0.78rem;color:#6d28d9">Assistant message ({len(content)} chars)</summary>'
                    f'<pre style="{_PRE};margin-top:0.3rem;border-left:3px solid #a78bfa">{_escape(content)}</pre>'
                    f'</details>'
                )
            if not body_parts:
                body_parts.append('<p style="margin:0.2rem 0;font-size:0.81rem;color:#94a3b8">Deciding next action…</p>')

        elif node == "tools":
            body_parts.extend(_tool_body(step, tool_inputs.get(i)))
        elif node == "a2a":
            body_parts.extend(_a2a_body(step, number))
        else:
            body_parts.append(
                f'<pre style="{_PRE}">{_escape(json.dumps(step, indent=2, default=str))}</pre>'
            )

        role_html = (
            f'<span style="font-size:0.74rem;color:#64748b">{role}</span>' if role else ""
        )
        footer = "" if prefix else (
            f'<div style="margin-top:0.4rem;font-size:0.67rem;color:#94a3b8;'
            f'border-top:1px dashed #e2e8f0;padding-top:0.3rem">≈ {cumulative:.2f}s elapsed in trace</div>'
        )
        card = (
            f'<div style="display:flex;background:{s["bg"]};border-radius:10px;'
            f'overflow:hidden;border:1px solid #e2e8f0;margin-bottom:0.6rem">'
            f'<div style="width:5px;background:{s["rail"]};flex-shrink:0"></div>'
            f'<div style="flex:1;padding:0.65rem 0.9rem 0.75rem;min-width:0">'
            # header row
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.4rem;margin-bottom:0.15rem">'
            f'<span style="font-size:0.63rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;font-weight:600">Step {number}</span>'
            f'<span style="font-size:0.66rem;font-weight:600;padding:0.1rem 0.45rem;border-radius:999px;'
            f'border:1px solid {s["rail"]};background:{s["pill_bg"]};color:{s["pill_fg"]};'
            f'text-transform:uppercase;letter-spacing:0.04em">{_escape(node)}</span>'
            f'{role_html}'
            f'<span style="margin-left:auto;font-weight:700;color:#b45309;font-size:0.8rem">{f"{lat:.2f}s" if step.get("latency_s") else ""}</span>'
            f'</div>'
            # title + subtitle
            f'<div style="font-size:0.93rem;font-weight:600;color:#1e293b;margin:0 0 0.1rem">{_escape(step.get("title") or s["title"])}</div>'
            f'<div style="font-size:0.77rem;color:#64748b;margin-bottom:0.4rem">{_escape(step.get("sub") or s["sub"])}</div>'
            # body
            + "".join(body_parts)
            + footer
            + f'</div></div>'
        )
        cards_html.append(card)
    return cards_html


def _lanes_html(lanes: List[Dict[str, Any]]) -> str:
    """One row per agent, each call placed at its wall-clock start, so delegations show as nested time."""
    spans = [(b["start"], b["end"]) for lane in lanes for b in lane["bars"]]
    if not spans:
        return ""
    t0 = min(a for a, _ in spans)
    total = max(b for _, b in spans) - t0 or 1e-9
    rows = []
    for lane in lanes:
        bars = []
        for b in lane["bars"]:
            colour = _STEP_STYLES[b["kind"]]["gantt"]
            left = 100 * (b["start"] - t0) / total
            width = max(0.5, 100 * (b["end"] - b["start"]) / total)
            fill = (
                f"background:repeating-linear-gradient(45deg,{colour}33,{colour}33 4px,{colour}11 4px,{colour}11 8px);"
                f"border:1px solid {colour}"
                if b["kind"] == "a2a"
                else f"background:{colour};opacity:0.85"
            )
            tip = _escape(f'{b["label"]} · {b["end"] - b["start"]:.1f}s')
            bars.append(
                f'<div title="{tip}" style="position:absolute;top:4px;bottom:4px;left:{left:.2f}%;'
                f'width:{width:.2f}%;border-radius:4px;{fill}"></div>'
            )
        rows.append(
            '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:3px">'
            f'<div style="width:11rem;flex-shrink:0;font-size:0.72rem;color:#475569;text-align:right;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{_escape(lane["agent"])}">{_escape(lane["agent"])}</div>'
            '<div style="position:relative;flex:1;height:22px;background:#f1f5f9;border-radius:6px;'
            'border:1px solid #e2e8f0;overflow:hidden">' + "".join(bars) + "</div></div>"
        )
    return (
        '<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#94a3b8;margin-bottom:0.25rem">'
        '<span>Timeline, one row per agent (hover a bar). Purple: model call. Teal: tool call. '
        'Hatched amber: the orchestrator waiting on an A2A task.</span>'
        f'<span>{total:.1f}s total</span></div>'
        + "".join(rows)
        + '<div style="margin-bottom:1rem"></div>'
    )


def visualize_agent_trace(result: dict) -> None:
    """Render the EVE agent trace - light theme, inline styles, GitHub-compatible.

    A step with ``node == "a2a"`` is one delegation to a specialist; its
    ``subtrace`` (the specialist's own agent and tool steps) is drawn inside it.
    ``result["lanes"]``, when present, is drawn as a timeline with one row per agent.
    """
    trace  = result.get("trace") or []
    query  = result.get("query", "")
    meta   = result.get("metadata") or {}
    lats   = (meta.get("latencies") or {})
    total_wall = float(lats.get("total_latency") or 0) or sum(
        float(s.get("latency_s") or 0) for s in trace
    ) or 1e-9

    if not trace:
        display(HTML("<p><em>No trace in this result.</em></p>"))
        return

    # ── Gantt bar ─────────────────────────────────────────────────────────────
    gantt_segs, t0 = [], 0.0
    for step in trace:
        lat   = float(step.get("latency_s") or 0)
        left  = 100 * t0 / total_wall
        width = max(0.6, 100 * lat / total_wall)
        s     = _STEP_STYLES.get((step.get("node") or "").lower(), _DEFAULT_STYLE)
        name  = step.get("name") or ((step.get("tool_calls") or [{}])[0].get("name")) or step.get("node", "?")
        tip   = _escape(f"{name} · {lat:.1f}s")
        gantt_segs.append(
            f'<div title="{tip}" style="position:absolute;top:4px;bottom:4px;'
            f'left:{left:.2f}%;width:{width:.2f}%;background:{s["gantt"]};'
            f'border-radius:5px;opacity:0.8"></div>'
        )
        t0 += lat

    gantt_html = "" if not lats else (
        '<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#94a3b8;margin-bottom:0.2rem">'
        f'<span>Timeline (by step latency)</span><span>{total_wall:.1f}s total</span></div>'
        '<div style="position:relative;height:26px;background:#f1f5f9;border-radius:7px;'
        'border:1px solid #e2e8f0;overflow:hidden;margin-bottom:1rem">'
        + "".join(gantt_segs) + "</div>"
    )
    lanes_html = _lanes_html(result.get("lanes") or [])

    # ── Legend: the step kinds that occur ─────────────────────────────────────
    present = {(s.get("node") or "").lower() for s in trace}
    legend_html = (
        '<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem">'
    )
    for label, s in _STEP_STYLES.items():
        if label not in present and label != "tools":
            continue
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem;'
            f'font-size:0.72rem;color:#475569">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
            f'background:{s["rail"]}"></span>{label}</span>'
        )
    legend_html += "</div>"

    summary = result.get("summary")
    summary_html = (
        f'<p style="margin:-0.5rem 0 0.85rem;font-size:0.8rem;color:#475569">{_escape(summary)}</p>' if summary else ""
    )

    html_doc = (
        '<div style="font-family:ui-sans-serif,system-ui,-apple-system,\'Segoe UI\',Roboto,Helvetica,Arial;'
        'background:#ffffff;color:#1e293b;border-radius:12px;padding:1.1rem 1.4rem 1.3rem;'
        'margin:0.5rem 0 1rem;border:1px solid #e2e8f0;box-shadow:0 1px 6px rgba(0,0,0,0.06)">'
        '<div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:0.2rem">Agent execution trace</div>'
        f'<p style="margin:0 0 0.85rem;font-size:0.84rem;color:#64748b">'
        f'<strong style="color:#1e293b">Query:</strong> {_escape(query)}</p>'
        + summary_html
        + legend_html
        + (lanes_html or gantt_html)
        + "".join(_cards(trace))
        + "</div>"
    )
    display(HTML(html_doc))


def _epoch(value: Any) -> float | None:
    if value is None:
        return None
    return value.timestamp() if hasattr(value, "timestamp") else float(value)


def _model_calls(sample: Any, asks: List[Dict[str, Any]]) -> Tuple[List[Any], Dict[str, List[Any]]]:
    """The sample's model calls: the top-level agent's, and each specialist's.

    Calls are grouped by their first input message (each agent's system prompt);
    the first call's group is the top-level agent's. Another group belongs to the
    specialist whose ``ask_*`` call was running when the group's first call started.
    A group that matches no ``ask_*`` call (a judge, say) is left out.
    """
    groups: Dict[str, List[Any]] = {}
    for event in sample.events:
        if getattr(event, "event", "") != "model":
            continue
        inputs = getattr(event, "input", None) or []
        key = str(inputs[0].content) if inputs else ""
        groups.setdefault(key, []).append(event)
    if not groups:
        return [], {}
    keys = list(groups)
    theirs: Dict[str, List[Any]] = {}
    for key in keys[1:]:
        start = _epoch(getattr(groups[key][0], "timestamp", None))
        for ask in asks:
            end = ask.get("ended_at") or float("inf")
            if start is not None and ask.get("started_at") and ask["started_at"] <= start <= end:
                theirs.setdefault(ask["artifact"]["agent"], []).extend(groups[key])
                break
    return groups[keys[0]], theirs


def _requests(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """A specialist's conversation, split at each incoming request (one per ``ask_*`` call)."""
    chunks: List[List[Dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "human":
            chunks.append([])
        elif chunks:
            chunks[-1].append(message)
    return chunks


def _specialist_steps(name: str, turns: List[Dict[str, Any]], models: List[Any], calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One request's turns as agent and tool steps; ``models`` and ``calls`` are consumed in order for latencies."""
    steps: List[Dict[str, Any]] = []
    for turn in turns:
        if turn.get("role") == "ai":
            event = models.pop(0) if models else None
            steps.append({
                "node": "agent",
                "role": "specialist",
                "title": f"{name} step",
                "sub": "The specialist plans its next tool call, or writes its reply to the orchestrator.",
                "content": turn.get("text") or "",
                "tool_calls": list(turn.get("tool_calls") or []),
                "latency_s": float(getattr(event, "working_time", 0) or 0),
            })
        elif turn.get("role") == "tool":
            call = calls.pop(0) if calls else {}
            latency = (call.get("ended_at") or 0) - (call.get("started_at") or 0)
            steps.append({
                "node": "tools",
                "role": "MCP",
                "name": turn.get("name"),
                "content": turn.get("text"),
                "latency_s": max(latency, 0.0),
            })
    return steps


def _lanes(name: str, own: List[Any], raw: List[Dict[str, Any]], subagents: Dict[str, Any], theirs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    def model_bars(events: List[Any], label: str) -> List[Dict[str, Any]]:
        bars = []
        for event in events:
            start, end = _epoch(getattr(event, "timestamp", None)), _epoch(getattr(event, "completed", None))
            if start is not None and end is not None:
                bars.append({"start": start, "end": end, "kind": "agent", "label": f"{label}: model call"})
        return bars

    def call_bars(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bars = []
        for entry in entries:
            if entry.get("started_at") and entry.get("ended_at"):
                kind = "a2a" if entry.get("artifact") else "tools"
                bars.append({"start": entry["started_at"], "end": entry["ended_at"], "kind": kind, "label": entry["name"]})
        return bars

    lanes = [{"agent": name, "bars": call_bars(raw) + model_bars(own, name)}]
    for slug, sub in subagents.items():
        bars = model_bars(theirs.get(slug, []), sub.get("name") or slug) + call_bars(sub.get("calls") or [])
        if bars:
            lanes.append({"agent": sub.get("name") or slug, "bars": bars})
    return lanes


def trace_from_sample(sample: Any) -> dict:
    """An Inspect sample, shaped like the EVE agentic response that ``visualize_agent_trace`` reads.

    Each assistant message is an agent step, with its tool calls and the
    model's wall time (``ModelEvent.working_time`` of the top-level agent's calls,
    in the same order). Each tool message is a tool step, with its output.

    For a multi-agent run (``langgraph_specialists``) each ``ask_*`` result is an
    ``a2a`` step instead: the A2A task from the raw trace, and as ``subtrace``
    the specialist's turns for that request, read from the sub-agent traces.
    ``lanes`` then holds each agent's model and tool calls on one clock.
    """
    from agentic_evals.agents.trace_callback import RAW_TRACE_KEY, SUBAGENT_TRACES_KEY

    store = getattr(sample, "store", None) or {}
    raw = list(store.get(RAW_TRACE_KEY) or [])
    subagents = dict(store.get(SUBAGENT_TRACES_KEY) or {})
    asks = [e for e in raw if (e.get("artifact") or {}).get("agent")]
    own, theirs = _model_calls(sample, asks)
    times = [float(e.working_time or 0.0) for e in own]

    pending = {slug: _requests(sub.get("messages") or []) for slug, sub in subagents.items()}
    models = {slug: list(theirs.get(slug, [])) for slug in subagents}
    calls = {slug: list(sub.get("calls") or []) for slug, sub in subagents.items()}
    queue = list(asks)

    query = next((m.text for m in sample.messages if m.role == "user"), "")
    stem = query.split("Question:", 1)[-1].strip()
    steps: list[dict[str, Any]] = []
    model_calls = 0
    for message in sample.messages:
        if message.role == "assistant":
            latency = times[model_calls] if model_calls < len(times) else 0.0
            model_calls += 1
            step = {
                "node": "agent",
                "role": "assistant",
                "content": message.text or "",
                "tool_calls": [{"name": t.function, "args": t.arguments} for t in (message.tool_calls or [])],
                "latency_s": latency,
            }
            if subagents:
                step.update(title="Orchestrator Step", sub="Plans the next delegation, or writes the final answer.")
            steps.append(step)
        elif message.role == "tool":
            ask = next((a for a in queue if a.get("name") == message.function), None)
            if ask is None:
                steps.append({"node": "tools", "role": "tool", "name": message.function, "content": message.text})
                continue
            queue.remove(ask)
            slug = ask["artifact"]["agent"]
            sub = subagents.get(slug) or {}
            turns = pending[slug].pop(0) if pending.get(slug) else []
            steps.append({
                "node": "a2a",
                "role": "peer",
                "name": message.function,
                "content": message.text,
                "agent": slug,
                "agent_name": sub.get("name") or slug,
                "a2a": ask["artifact"],
                "request": (ask.get("input") or {}).get("request", ""),
                "latency_s": max((ask.get("ended_at") or 0) - (ask.get("started_at") or 0), 0.0),
                "subtrace": _specialist_steps(sub.get("name") or slug, turns, models.get(slug, []), calls.get(slug, [])),
            })
    result: dict[str, Any] = {"query": stem[:400], "trace": steps, "metadata": {}}
    if subagents:
        used = [s for s in subagents if subagents[s].get("context_ids")]
        result["summary"] = (
            f"Orchestrator with {len(subagents)} specialist agents reachable over A2A; "
            f"{len(asks)} A2A task(s) sent, to {', '.join(subagents[s].get('name') or s for s in used) or 'none'}."
        )
        result["lanes"] = _lanes("Orchestrator", own, raw, {s: subagents[s] for s in used}, theirs)
    return result


def show_trace(log: Any, question: int | str) -> None:
    """Render one sample of an Inspect log as the agent trace, with any specialist runs nested in it."""
    sample = next(s for s in log.samples if int(s.id) == int(question))
    visualize_agent_trace(trace_from_sample(sample))
