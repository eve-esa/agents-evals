"""Write each specialist's A2A self-description and system prompt with a chat model.

Per specialist, one call returns one JSON object:

- ``name`` and ``description``: the AgentCard's human-readable name and its
  description (A2A spec 1.0, section 4.4.1).
- ``skills``: 1 to 3 AgentSkills (section 4.4.5), each a task an orchestrator
  would delegate, with ``id``, ``name``, ``description``, ``tags`` and
  ``examples``. A skill may also list the owned ``tools`` it relies on. That
  list is an internal hint, not an A2A field: it never reaches the card.
- ``role_prompt``: the start of the specialist's own system prompt. The A2A
  the one input rule the protocol needs is added in code (``graph.INPUT_PROMPT``).

``DESCRIBE_SYSTEM`` quotes the spec's field definitions and one example from
a2a-samples (airbnb_planner_multiagent) whose form the model copies. The reply
is validated; on any problem the model is asked once more with the problems
listed. ``slug``, the identifier of the ``ask_<slug>`` tool and the URL key, is
derived from ``name``.

Placeholders (``placeholder_specialist``) give the same shape with no model
call: the group label as name, one skill per tool, no role prompt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ToolFields = tuple[str, str, list[str]]

_SLUG = re.compile(r"^[a-z][a-z0-9_]*$")
SLUG_CHARS = 40
NAME_CHARS = 60
MAX_SKILLS = 3
MAX_EXAMPLES = 3
MIN_ROLE_WORDS = 100
MAX_ROLE_WORDS = 400
_SKILL_TEXT_CHARS = 300


@dataclass(frozen=True)
class DescriberConfig:
    """The chat model that writes the A2A text, from the ``describer`` block of ``embedding.yaml``."""

    base_url: str
    model: str
    api_key_env: str


def load_describer_config(path: Path) -> DescriberConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = data.get("describer") or {}
    missing = [f for f in ("base_url", "model", "api_key_env") if not block.get(f)]
    if missing:
        raise ValueError(f"{path}: describer is missing {', '.join(missing)}")
    return DescriberConfig(
        base_url=str(block["base_url"]), model=str(block["model"]), api_key_env=str(block["api_key_env"])
    )


def first_line(text: str, limit: int = _SKILL_TEXT_CHARS) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def tool_skill(tool: str, description: str, tag: str) -> dict[str, Any]:
    """One AgentSkill for one tool: the shape placeholders and old plan files fall back to."""
    return {
        "id": tool,
        "name": tool,
        "description": first_line(description) or tool,
        "tags": [tag],
        "examples": [],
        "tools": [tool],
    }


def placeholder_specialist(label: str, tools: list[ToolFields]) -> dict[str, Any]:
    """A specialist entry with no model call: the label as name, one skill per tool."""
    names = [name for name, _, _ in tools]
    return {
        "slug": label,
        "name": label,
        "description": f"Owns {len(names)} tools: {', '.join(names)}.",
        "tools": names,
        "skills": [tool_skill(name, description, label) for name, description, _ in tools],
        "role_prompt": "",
    }


# The worked example. Its form is copied; its content is not.
A2A_SAMPLE_EXAMPLE = {
    "name": "Airbnb Agent",
    "description": "Helps with searching accommodation on Airbnb and answering questions about the listings it finds.",
    "skills": [
        {
            "id": "airbnb_search",
            "name": "Search airbnb accommodation",
            "description": "Helps with accommodation search using airbnb: finds listings for a location, dates and number of guests, and reports prices and links.",
            "tags": ["airbnb accommodation", "lodging search"],
            "examples": ["Please find a room in LA, CA, April 15, 2025, checkout date is april 18, 2 adults"],
        }
    ],
    "role_prompt": "## Role and scope\nYou are a specialized assistant for Airbnb accommodations. You search Airbnb listings and answer questions about them. If a request is about anything else, such as flights or the weather, say that you only handle accommodation and that another agent should take it.\n\n## Required inputs\nA search needs a location, a check-in date, a check-out date and the number of guests. Check these in that order before you search. A question about a listing needs the listing id or the link from an earlier search.\n\n## How to work\nUse the search tools to find listings that match every input, then fetch the details of the best matches. Rely only on what the tools return: never invent listings, prices or availability. If a search returns nothing, widen it once, to a nearby area or flexible dates, and say that you did.\n\n## What to report\nList each listing with its name, price per night and total, rating and direct link. Include all relevant tool output, and say plainly what failed or could not be found.",
}

DESCRIBE_SYSTEM = f"""\
You write the A2A (Agent2Agent protocol) self-description of one specialist \
agent, and the opening of its own system prompt.

The A2A spec (1.0) defines the fields this way. Use these meanings:
- AgentCard name: "A human readable name for the agent."
- AgentCard description: "A human-readable description of the agent, assisting users and other agents in understanding its purpose."
- AgentSkill id: "A unique identifier for the agent's skill."
- AgentSkill name: "A human-readable name for the skill."
- AgentSkill description: "A detailed description of the skill."
- AgentSkill tags: "A set of keywords describing the skill's capabilities."
- AgentSkill examples: "Example prompts or scenarios that this skill can handle."

Who reads what:
- An orchestrator agent reads the card (name, description) and the skills, to pick this agent for a step. It then writes one standalone request message, guided by the skill examples. The specialist sees nothing else, so that message must carry every input (paths, bands, thresholds, dates, column names).
- The specialist reads role_prompt as the start of its own system prompt.

House style (from a2a-samples):
- Card name: 2 to 4 words, ending in "Agent".
- Card description: 1 or 2 sentences.
- Skills: 1 to 3. Each skill is a TASK an orchestrator would delegate, for example "Compute spectral indices from band rasters". Never write one skill per tool. A skill must not name tools or describe them one by one: one skill covers several tools.
- Skill description: 1 or 2 sentences saying what the task does and what a request must carry.
- Skill tags: 2 to 5 lowercase keywords.
- Skill examples: 1 to 3 realistic requests that carry every input, with placeholder paths such as data/scene_B4.tif.
- Skill tools (optional): the names of the owned tools this skill relies on. It is an internal hint and is not shown on the card. Leave it out when unsure.

role_prompt: 150 to 350 words, in the second person ("You ..."), with these four markdown sections, in this order:
## Role and scope: what the specialist handles. It declines out-of-scope requests and says which kind of agent should take them.
## Required inputs: for each kind of request, what must be present (paths, bands, thresholds, dates, column names). These are checked in order before any tool runs.
## How to work: which tools to use, in what order, described as tasks. Check each output. Say what to try when a result is empty or a call fails.
## What to report: values with their units, files written, and failures.
role_prompt does not describe the reply format or the protocol. Those are added separately.

Use only facts from the tool descriptions. If they are unclear, say less rather than guess.

Return one JSON object and nothing else. Its keys are those of the example below: name, description, skills (each with id, name, description, tags, examples and optionally tools), and role_prompt.

Copy the form of this example, not its content. It is from a2a-samples/samples/python/agents/airbnb_planner_multiagent (card and skill as published; role_prompt expands its system instruction into the four sections):
{json.dumps(A2A_SAMPLE_EXAMPLE, indent=2)}"""


def describe_user(tools: list[ToolFields], others: list[list[str]]) -> str:
    """The user message: each tool this agent owns with its full description, and the tool names other agents cover.

    The whole description is sent, not a first line, so the card can describe what the tools really do.
    Continuation lines are indented under their tool. No internal group labels appear here.
    """
    lines = ["Tools it owns:"]
    for tool, description, args in tools:
        arg_text = f" (args: {', '.join(args)})" if args else ""
        text = (description or "").strip() or "(no description)"
        body = text.replace("\n", "\n    ")
        lines.append(f"- {tool}{arg_text}:\n    {body}")
    if others:
        lines += ["", "Other agents in this set cover (for what this agent does not do):"]
        lines += [f"- Another agent covers: {', '.join(names[:8])}" for names in others]
    return "\n".join(lines)


def parse_json(text: str) -> Any:
    """The JSON object in a reply, also when it sits in a code fence or between prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _texts(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [t for t in (_text(v) for v in values) if t]


def validate(raw: Any, owned: list[str]) -> tuple[dict[str, Any], list[str]]:
    """``(clean entry, problems)`` for one describer reply.

    ``role_prompt`` keeps its newlines (it has markdown headings). Skill ``tools``
    keep only the owned tools; a skill with none has no ``tools`` key, so it is
    kept for every sample.
    """
    if not isinstance(raw, dict):
        return {}, ["the reply is not a JSON object"]
    problems: list[str] = []
    name = _text(raw.get("name"))
    description = _text(raw.get("description"))
    role_prompt = str(raw.get("role_prompt") or "").strip()
    if not name or len(name) > NAME_CHARS:
        problems.append(f"name must be 1 to {NAME_CHARS} characters")
    if not description:
        problems.append("description must not be empty")
    words = len(role_prompt.split())
    if not MIN_ROLE_WORDS <= words <= MAX_ROLE_WORDS:
        problems.append(f"role_prompt must be {MIN_ROLE_WORDS} to {MAX_ROLE_WORDS} words, got {words}")
    raw_skills = raw.get("skills")
    if not isinstance(raw_skills, list):
        raw_skills = []
    if not 1 <= len(raw_skills) <= MAX_SKILLS:
        problems.append(f"skills must be a list of 1 to {MAX_SKILLS} skills, got {len(raw_skills)}")
    skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, s in enumerate(raw_skills, 1):
        if not isinstance(s, dict):
            problems.append(f"skill {i} is not an object")
            continue
        sid = _text(s.get("id"))
        where = f"skill {sid or i}"
        if not _SLUG.fullmatch(sid):
            problems.append(f"{where}: id must be snake_case")
        elif sid in seen:
            problems.append(f"{where}: id is repeated")
        seen.add(sid)
        entry: dict[str, Any] = {
            "id": sid,
            "name": _text(s.get("name")),
            "description": _text(s.get("description")),
            "tags": _texts(s.get("tags")),
            "examples": _texts(s.get("examples")),
        }
        for key in ("name", "description", "tags"):
            if not entry[key]:
                problems.append(f"{where}: {key} must not be empty")
        if not 1 <= len(entry["examples"]) <= MAX_EXAMPLES:
            problems.append(f"{where}: examples must be 1 to {MAX_EXAMPLES} requests")
        tools = [t for t in _texts(s.get("tools")) if t in owned]
        if tools:
            entry["tools"] = tools
        skills.append(entry)
    clean = {"name": name, "description": description, "skills": skills, "role_prompt": role_prompt}
    return clean, problems


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = re.sub(r"_agent$", "", slug) or "agent"
    if not slug[0].isalpha():
        slug = f"agent_{slug}"
    return slug[:SLUG_CHARS].rstrip("_")


def _unique(slug: str, taken: set[str]) -> str:
    candidate, n = slug, 2
    while candidate in taken:
        suffix = f"_{n}"
        candidate, n = slug[: SLUG_CHARS - len(suffix)] + suffix, n + 1
    taken.add(candidate)
    return candidate


async def _chat_complete(config: DescriberConfig, system: str, user: str) -> str:
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.api_key_env} is not set; put it in .env (named by describer.api_key_env)")
    llm = ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=0,
        max_retries=3,
        # Same switch the Earth-Bench eval configs use: no reasoning tokens.
        extra_body={"reasoning": {"enabled": False}},
    )
    reply = await llm.ainvoke([("system", system), ("user", user)])
    return str(reply.content)


def _retry_note(problems: list[str]) -> str:
    return "\n\nYour last answer had these problems. Return the corrected JSON object:\n" + "\n".join(
        f"- {item}" for item in problems
    )


async def _describe_one(
    spec: dict[str, Any], by_name: dict[str, ToolFields], others: list[list[str]], config: DescriberConfig
) -> dict[str, Any]:
    owned = list(spec["tools"])
    user = describe_user([by_name[t] for t in owned], others)
    reply = await _chat_complete(config, DESCRIBE_SYSTEM, user)
    clean, problems = _parsed(reply, owned)
    if problems:
        reply = await _chat_complete(config, DESCRIBE_SYSTEM, user + _retry_note(problems))
        clean, problems = _parsed(reply, owned)
    if problems:
        raise ValueError(f"the describer's answer for {spec['slug']} is still invalid: {'; '.join(problems)}")
    return clean


def _parsed(reply: str, owned: list[str]) -> tuple[dict[str, Any], list[str]]:
    try:
        raw = parse_json(reply)
    except json.JSONDecodeError:
        return {}, ["the reply is not valid JSON"]
    return validate(raw, owned)


async def describe_specialists(
    plan: dict[str, Any], fields: list[ToolFields], config: DescriberConfig
) -> dict[str, Any]:
    """Fill each specialist's slug, name, description, skills and role prompt from the chat model."""
    by_name = {name: (name, description, args) for name, description, args in fields}
    specialists = plan["specialists"]

    def others(spec: dict[str, Any]) -> list[list[str]]:
        return [list(o["tools"]) for o in specialists if o is not spec]

    described = await asyncio.gather(*(_describe_one(s, by_name, others(s), config) for s in specialists))
    taken: set[str] = set()
    for spec, clean in zip(specialists, described):
        spec.update(clean)
        spec["slug"] = _unique(slugify(clean["name"]), taken)
    plan["describer_model"] = config.model
    return plan
