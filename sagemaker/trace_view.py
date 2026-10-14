"""Agent trace view for the workshop notebook.

The renderer ``visualize_agent_trace`` is taken from the EVE notebook
``Agentic_EVE.ipynb`` in https://github.com/eve-esa/phinnovation (cell "Tracing").
Changes from that notebook: the timeline bar is drawn only when the caller
gives latencies (Inspect does not time tool calls), and the adapter
``trace_from_sample`` turns an Inspect sample into the shape it reads.
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
        elif node == "tools":
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
}
_DEFAULT_STYLE = {
    "bg": "#f8fafc", "rail": "#64748b",
    "pill_bg": "#f1f5f9", "pill_fg": "#475569",
    "title": "Step", "sub": "", "gantt": "#94a3b8",
}

# Reusable inline style strings
_PRE  = "margin:0;padding:0.45rem 0.65rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:0 6px 6px 0;font-size:0.7rem;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#334155;font-family:ui-monospace,'SFMono-Regular',Consolas,monospace"
_LBL  = "font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#64748b;display:block;margin:0.55rem 0 0.2rem"
_CODE = "background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:0.78rem;color:#1e293b;font-family:ui-monospace,'SFMono-Regular',Consolas,monospace"


def visualize_agent_trace(result: dict) -> None:
    """Render the EVE agent trace - light theme, inline styles, GitHub-compatible."""
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

    tool_inputs = _pair_tool_inputs(trace)

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

    # ── Step cards ────────────────────────────────────────────────────────────
    cards_html, cumulative = [], 0.0

    for i, step in enumerate(trace):
        node = (step.get("node") or "step").lower()
        s    = _STEP_STYLES.get(node, _DEFAULT_STYLE)
        lat  = float(step.get("latency_s") or 0)
        cumulative += lat
        role = _escape(step.get("role", ""))

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
            tname = _escape(step.get("name", "tool"))
            body_parts.append(
                f'<p style="margin:0 0 0.3rem;font-size:0.84rem;color:#1e293b">Tool: '
                f'<code style="background:#cffafe;color:#155e75;padding:2px 8px;'
                f'border-radius:5px;font-size:0.8rem;font-family:ui-monospace,monospace">{tname}</code></p>'
            )
            inp = tool_inputs.get(i)
            if inp is not None:
                try:
                    inp_txt = json.dumps(inp, indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    inp_txt = str(inp)
                body_parts.append(
                    f'<span style="{_LBL}">Input - arguments passed to the tool</span>'
                    f'<pre style="{_PRE};border-left:3px solid #7c3aed">{_escape(inp_txt)}</pre>'
                )
            raw = step.get("content")
            out_txt, is_json = _extract_tool_output(raw)
            out_body = _highlight_json(out_txt) if is_json else _escape(out_txt)
            body_parts.append(
                f'<span style="{_LBL}">Output - value returned to the model</span>'
                f'<pre style="{_PRE};border-left:3px solid #0891b2">{out_body}</pre>'
            )
        else:
            body_parts.append(
                f'<pre style="{_PRE}">{_escape(json.dumps(step, indent=2, default=str))}</pre>'
            )

        role_html = (
            f'<span style="font-size:0.74rem;color:#64748b">{role}</span>' if role else ""
        )
        card = (
            f'<div style="display:flex;background:{s["bg"]};border-radius:10px;'
            f'overflow:hidden;border:1px solid #e2e8f0;margin-bottom:0.6rem">'
            f'<div style="width:5px;background:{s["rail"]};flex-shrink:0"></div>'
            f'<div style="flex:1;padding:0.65rem 0.9rem 0.75rem;min-width:0">'
            # header row
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.4rem;margin-bottom:0.15rem">'
            f'<span style="font-size:0.63rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;font-weight:600">Step {i+1}</span>'
            f'<span style="font-size:0.66rem;font-weight:600;padding:0.1rem 0.45rem;border-radius:999px;'
            f'border:1px solid {s["rail"]};background:{s["pill_bg"]};color:{s["pill_fg"]};'
            f'text-transform:uppercase;letter-spacing:0.04em">{_escape(node)}</span>'
            f'{role_html}'
            f'<span style="margin-left:auto;font-weight:700;color:#b45309;font-size:0.8rem">{f"{lat:.2f}s" if step.get("latency_s") else ""}</span>'
            f'</div>'
            # title + subtitle
            f'<div style="font-size:0.93rem;font-weight:600;color:#1e293b;margin:0 0 0.1rem">{s["title"]}</div>'
            f'<div style="font-size:0.77rem;color:#64748b;margin-bottom:0.4rem">{s["sub"]}</div>'
            # body
            + "".join(body_parts)
            # footer
            + f'<div style="margin-top:0.4rem;font-size:0.67rem;color:#94a3b8;'
            f'border-top:1px dashed #e2e8f0;padding-top:0.3rem">≈ {cumulative:.2f}s elapsed in trace</div>'
            f'</div></div>'
        )
        cards_html.append(card)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = (
        '<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem">'
    )
    for label, s in _STEP_STYLES.items():
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem;'
            f'font-size:0.72rem;color:#475569">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
            f'background:{s["rail"]}"></span>{label}</span>'
        )
    legend_html += "</div>"

    html_doc = (
        '<div style="font-family:ui-sans-serif,system-ui,-apple-system,\'Segoe UI\',Roboto,Helvetica,Arial;'
        'background:#ffffff;color:#1e293b;border-radius:12px;padding:1.1rem 1.4rem 1.3rem;'
        'margin:0.5rem 0 1rem;border:1px solid #e2e8f0;box-shadow:0 1px 6px rgba(0,0,0,0.06)">'
        '<div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:0.2rem">Agent execution trace</div>'
        f'<p style="margin:0 0 0.85rem;font-size:0.84rem;color:#64748b">'
        f'<strong style="color:#1e293b">Query:</strong> {_escape(query)}</p>'
        + legend_html
        + gantt_html
        + "".join(cards_html)
        + "</div>"
    )
    display(HTML(html_doc))


def trace_from_sample(sample: Any) -> dict:
    """An Inspect sample, shaped like the EVE agentic response that ``visualize_agent_trace`` reads.

    Each assistant message is an agent step, with its tool calls and the
    model's wall time (``ModelEvent.working_time``, in the same order). Each tool
    message is a tool step, with its output.
    """
    times = [float(e.working_time or 0.0) for e in sample.events if getattr(e, "event", "") == "model"]
    query = next((m.text for m in sample.messages if m.role == "user"), "")
    stem = query.split("Question:", 1)[-1].strip()
    steps: list[dict[str, Any]] = []
    model_calls = 0
    for message in sample.messages:
        if message.role == "assistant":
            latency = times[model_calls] if model_calls < len(times) else 0.0
            model_calls += 1
            steps.append({
                "node": "agent",
                "role": "assistant",
                "content": message.text or "",
                "tool_calls": [{"name": t.function, "args": t.arguments} for t in (message.tool_calls or [])],
                "latency_s": latency,
            })
        elif message.role == "tool":
            steps.append({"node": "tools", "role": "tool", "name": message.function, "content": message.text})
    return {"query": stem[:400], "trace": steps, "metadata": {}}


def show_trace(log: Any, question: int | str) -> None:
    """Render one sample of an Inspect log as the agent trace."""
    sample = next(s for s in log.samples if int(s.id) == int(question))
    visualize_agent_trace(trace_from_sample(sample))
