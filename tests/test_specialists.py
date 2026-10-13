"""Specialists solver: splitting tools, the describer, the committed-plan cache, and A2A."""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import sys

import httpx
import numpy as np
import pytest
from a2a.client import A2ACardResolver
from langchain_core.tools import StructuredTool

from agentic_evals.agents.specialists import clustering, describer
from agentic_evals.agents.specialists.a2a import (
    SubAgentExecutor,
    agent_card,
    ask_tools,
    serve,
)
from agentic_evals.agents.specialists.clustering import (
    build_file_plan,
    build_plan,
    launched_files,
    mcp_server_files,
    plan_specialists,
    task_tool_dir,
    write_plan,
)
from agentic_evals.agents.specialists.describer import (
    DescriberConfig,
    describe_specialists,
)
from agentic_evals.agents.trace_callback import TraceRecorder

# Three themes; the fake embedder puts each on its own axis.
TOOLS = [
    ("calc_ndvi", "Compute the NDVI vegetation index from red and nir bands.", ["red", "nir"]),
    ("calc_evi", "Compute the EVI vegetation index from bands.", ["blue", "red", "nir"]),
    ("calc_savi", "Compute the SAVI soil adjusted vegetation index.", ["red", "nir"]),
    ("trend_test", "Mann-Kendall statistical trend test on a time series.", ["series"]),
    ("change_point", "Detect a change point in a time series statistically.", ["series"]),
    ("mean_series", "Statistical mean of a time series.", ["series"]),
    ("detect_ships", "Detect ships in an RGB image with an object detector.", ["image"]),
    ("count_cars", "Count cars in an RGB image with an object detector.", ["image"]),
    ("segment_water", "Segment water in an RGB image.", ["image"]),
]
CONFIG = DescriberConfig(base_url="http://x", model="m", api_key_env="K")


def _theme(text: str) -> int:
    lowered = text.lower()
    return 0 if "vegetation" in lowered else 1 if "series" in lowered else 2


async def fake_embed(texts: list[str], config) -> np.ndarray:
    rng = np.random.default_rng(0)
    vectors = np.eye(3)[[_theme(t) for t in texts]] + 0.01 * rng.standard_normal((len(texts), 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _plan(**kwargs):
    return asyncio.run(build_plan(TOOLS, embedder=fake_embed, **kwargs))


def _groups(plan):
    return sorted(sorted(s["tools"]) for s in plan["specialists"])


EXPECTED = sorted(
    [
        ["calc_evi", "calc_ndvi", "calc_savi"],
        ["change_point", "mean_series", "trend_test"],
        ["count_cars", "detect_ships", "segment_water"],
    ]
)


@pytest.mark.parametrize("algorithm", ["ward", "kmeans"])
def test_clusters_follow_the_themes(algorithm):
    plan = _plan(n_agents=3, algorithm=algorithm)
    assert plan["k"] == 3 and _groups(plan) == EXPECTED
    for s in plan["specialists"]:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", f"ask_{s['slug']}")
        assert [skill["tools"] for skill in s["skills"]] == [[t] for t in s["tools"]]  # placeholder: one per tool
    assert plan == _plan(n_agents=3, algorithm=algorithm)  # deterministic


def test_k_from_tools_per_agent_and_auto():
    assert _plan(tools_per_agent=3)["k"] == 3
    assert _plan()["k"] == 1  # 9 tools, 9 per agent
    assert _plan(n_agents="auto", max_agents=5)["k"] == 3
    assert _plan(n_agents=50)["k"] == len(TOOLS)  # capped at one tool per agent


def test_labels_name_the_theme():
    assert [s["slug"] for s in _plan(n_agents=3)["specialists"]] == ["1_red_nir", "2_series_time", "3_image_rgb"]


def _tools(subset=None):
    return [
        StructuredTool.from_function(func=lambda **_: "", name=name, description=description)
        for name, description, _ in TOOLS
        if subset is None or name in subset
    ]


async def _no_embed(*_args, **_kwargs):
    raise AssertionError("the embeddings API must not be called when a committed plan covers the tools")


# The describer: fake chat replies built from the prompt's tool list.


def _owned(user: str) -> list[str]:
    section = user.split("Tools it owns:\n", 1)[1].split("\n\n", 1)[0]
    return [line[2:].split(" ", 1)[0].rstrip(":") for line in section.splitlines() if line.startswith("- ")]


ROLE_PROMPT = """\
## Role and scope
You are a raster specialist for vegetation and soil indices. You compute index values from band rasters that the caller names by path. If a request is about statistics, classification or anything else, say that you only handle these indices and that another agent should take it.

## Required inputs
Check the inputs in this order before you call any tool. First, the path of every band raster the index needs, such as red and near-infrared. Second, the name of the index. Third, the output path if the caller wants a file written. If a path is missing, ask for it instead of guessing.

## How to work
Use the index tools that match the request, one index at a time. Check each output: it must report a value and a file path. If a tool fails because a band is missing, name the band that is missing and stop. If a result is empty, check that the bands cover the same extent before you try again.

## What to report
Report each index value with its unit, the file written, and any band or step that failed. Say plainly when a value could not be computed."""


def good_reply(user: str, name: str | None = None) -> str:
    owned = _owned(user)
    return json.dumps(
        {
            "name": name or f"{owned[0].replace('_', ' ').title()} Agent",
            "description": f"Runs {', '.join(owned)} on rasters and returns the values.",
            "skills": [
                {
                    "id": "main_skill",
                    "name": "Main skill",
                    "description": "Computes the values; send the input file paths.",
                    "tags": ["raster"],
                    "examples": ["Compute it for data/a.tif."],
                    "tools": owned,
                }
            ],
            "role_prompt": ROLE_PROMPT,
        }
    )


def _describe(monkeypatch, chat, plan=None):
    monkeypatch.setattr(describer, "_chat_complete", chat)
    return asyncio.run(describe_specialists(plan or _plan(n_agents=3), list(TOOLS), CONFIG))


def test_describer_fills_every_a2a_field(monkeypatch):
    calls = []

    async def chat(config, system, user):
        calls.append(user)
        return good_reply(user)

    plan = _describe(monkeypatch, chat)
    first = plan["specialists"][0]
    assert first["name"] == "Calc Evi Agent" and first["slug"] == "calc_evi"
    assert first["skills"][0]["tags"] == ["raster"] and first["skills"][0]["tools"] == first["tools"]
    assert first["role_prompt"].startswith("## Role and scope")
    assert plan["describer_model"] == "m"
    assert "Tools it owns:" in calls[0] and "(args: blue, red, nir)" in calls[0]
    # The other groups are listed by their tool names, so the card can say what it does not do.
    assert "Another agent covers: change_point, mean_series, trend_test" in calls[0]
    # No internal group labels reach the prompt.
    assert "2_series_time" not in calls[0] and "1_red_nir" not in calls[0] and "3_image_rgb" not in calls[0]


def test_describer_retries_with_the_problems_listed(monkeypatch):
    seen = []

    async def chat(config, system, user):
        seen.append(user)
        if len(seen) == 1:  # first specialist, first answer: bad skill id, 20-word role prompt
            reply = json.loads(good_reply(user))
            reply["skills"][0]["id"] = "Main Skill"
            reply["role_prompt"] = (
                "You are a raster specialist who checks every path before running a tool and reports each value with its unit."
            )
            return json.dumps(reply)
        return good_reply(user)

    plan = _describe(monkeypatch, chat)
    retry = [u for u in seen if "had these problems" in u]
    assert len(retry) == 1
    assert "id must be snake_case" in retry[0] and "role_prompt must be 100 to 400 words, got 20" in retry[0]
    assert plan["specialists"][0]["skills"][0]["id"] == "main_skill"


def test_skill_tools_are_trimmed_to_owned(monkeypatch):
    async def chat(config, system, user):
        reply = json.loads(good_reply(user))
        owned = _owned(user)
        base = reply["skills"][0]
        reply["skills"] = [
            {**base, "id": "first_skill", "tools": [owned[0], "not_a_tool"]},
            {**base, "id": "second_skill", "tools": ["ghost_tool"]},
        ]
        return json.dumps(reply)

    plan = _describe(monkeypatch, chat)
    first, second = plan["specialists"][0]["skills"]
    assert first["tools"] == [plan["specialists"][0]["tools"][0]]  # calc_evi: the first owned tool
    assert "tools" not in second  # no owned tool left: the key is omitted, so _restrict keeps the skill


def test_more_than_three_skills_is_rejected(monkeypatch):
    async def chat(config, system, user):
        reply = json.loads(good_reply(user))
        base = reply["skills"][0]
        reply["skills"] = [{**base, "id": f"skill_{i}"} for i in range(4)]
        return json.dumps(reply)

    with pytest.raises(ValueError, match="still invalid"):
        _describe(monkeypatch, chat)


def test_still_invalid_after_the_retry_raises(monkeypatch):
    async def chat(config, system, user):
        return "not json at all"

    with pytest.raises(ValueError, match="still invalid"):
        _describe(monkeypatch, chat)


def test_json_in_prose_is_parsed(monkeypatch):
    seen = []

    async def chat(config, system, user):
        seen.append(user)
        reply = good_reply(user)
        if len(seen) == 1:
            return f"Sure:\n```json\n{reply}\n```"
        return f"Here it is: {reply} Done."

    plan = _describe(monkeypatch, chat)
    assert len(seen) == len(plan["specialists"])  # one call each: no retry
    assert plan["specialists"][0]["name"] == "Calc Evi Agent"


def test_duplicate_names_get_unique_slugs(monkeypatch):
    async def chat(config, system, user):
        return good_reply(user, name="Raster Agent")

    slugs = [s["slug"] for s in _describe(monkeypatch, chat)["specialists"]]
    assert slugs == ["raster", "raster_2", "raster_3"]


def test_prompt_quotes_the_spec_and_the_sample(monkeypatch):
    system_prompts = []

    async def chat(config, system, user):
        system_prompts.append(system)
        return good_reply(user)

    plan = _describe(monkeypatch, chat)
    system = describer.DESCRIBE_SYSTEM
    assert system_prompts[0] == system
    assert "A detailed description of the skill." in system
    assert "Example prompts or scenarios that this skill can handle." in system
    assert "Airbnb Agent" in system
    assert plan["specialists"][0]["role_prompt"] == ROLE_PROMPT  # newlines and headings kept


# The committed-plan cache.


def test_committed_plan_is_used_without_embedding(tmp_path, monkeypatch):
    async def chat(config, system, user):
        return good_reply(user)

    write_plan(_describe(monkeypatch, chat), tmp_path / "bench.json")
    monkeypatch.setattr(clustering, "embed", _no_embed)
    specialists = asyncio.run(plan_specialists(_tools(), directory=tmp_path))
    assert sorted(sorted(s.tools) for s in specialists) == EXPECTED
    assert specialists[0].slug == "calc_evi" and specialists[0].role_prompt
    # A sample with a subset of the tools (EO-Gym) gets the same split, restricted; skills follow.
    subset = asyncio.run(plan_specialists(_tools({"calc_ndvi", "trend_test"}), directory=tmp_path))
    assert [s.tools for s in subset] == [("calc_ndvi",), ("trend_test",)]
    assert [skill["tools"] for skill in subset[0].skills] == [["calc_ndvi"]]


def test_restrict_drops_skills_left_empty(tmp_path):
    plan = _plan(n_agents=3)  # placeholder: one skill per tool
    write_plan(plan, tmp_path / "bench.json")
    (veg,) = asyncio.run(plan_specialists(_tools({"calc_ndvi"}), directory=tmp_path))
    assert [skill["id"] for skill in veg.skills] == ["calc_ndvi"]


def test_changed_tools_need_a_new_plan(tmp_path, monkeypatch):
    plan = _plan(n_agents=3)
    plan["tools"]["calc_ndvi"] = "stale"
    write_plan(plan, tmp_path / "bench.json")
    monkeypatch.setattr(clustering, "embed", _no_embed)
    with pytest.raises(LookupError, match="specialists.clustering --task"):
        asyncio.run(plan_specialists(_tools(), directory=tmp_path))


def test_recompute_writes_the_plan_once(tmp_path, monkeypatch):
    async def chat(config, system, user):
        return good_reply(user)

    monkeypatch.setattr(clustering, "embed", fake_embed)
    monkeypatch.setattr(describer, "_chat_complete", chat)
    first = asyncio.run(plan_specialists(_tools(), recompute=True, directory=tmp_path))
    written = list(tmp_path.glob("auto_*.json"))
    assert len(written) == 1 and json.loads(written[0].read_text())["k"] == 1
    monkeypatch.setattr(clustering, "embed", _no_embed)
    assert asyncio.run(plan_specialists(_tools(), directory=tmp_path)) == first


def test_bad_plan_is_rejected(tmp_path):
    plan = _plan(n_agents=3)
    plan["specialists"][0]["tools"].append("calc_evi")
    with pytest.raises(ValueError, match="exactly one specialist"):
        write_plan(plan, tmp_path / "bad.json")


# The files algorithm.

FASTMCP_FILE = '''
from fastmcp import FastMCP
mcp = FastMCP()

@mcp.tool(description="NDVI.")
def calc_ndvi(red, nir): ...

@mcp.tool(name="calc_evi", description="EVI.")
def evi_impl(blue, red, nir): ...

@mcp.tool
def calc_savi(red, nir): ...

def helper(): ...
'''

LOWLEVEL_FILE = '''
import mcp.server.stdio
from mcp.server.lowlevel import Server

def build():
    server: Server = Server("stats")

    @server.list_tools()
    async def list_tools(): ...
    return server
'''

NOT_A_SERVER = '''
import runpy, sys
from fastmcp import Context  # an MCP import, but no server is built here
runpy.run_path(sys.argv[1])
'''


def _tool_repo(tmp_path):
    (tmp_path / "Index.py").write_text(FASTMCP_FILE)
    (tmp_path / "stats_server.py").write_text(LOWLEVEL_FILE)
    (tmp_path / "guard.py").write_text(NOT_A_SERVER)
    (tmp_path / "__init__.py").write_text("")
    return tmp_path


def _fields(*names):
    by_name = {t[0]: t for t in TOOLS}
    return [by_name[n] for n in names]


def test_server_files_are_found_by_their_mcp_server(tmp_path):
    found = {f.path.name: f for f in mcp_server_files(_tool_repo(tmp_path))}
    assert sorted(found) == ["Index.py", "stats_server.py"]  # guard.py and __init__.py build no server
    assert found["Index.py"].tools == ("calc_ndvi", "calc_evi", "calc_savi")  # name= wins over the function name
    assert found["stats_server.py"].tools == () and found["stats_server.py"].server_vars == ("server",)


def test_file_plan_assigns_static_and_launched_tools(tmp_path):
    repo = _tool_repo(tmp_path)
    servers = {
        # Like Earth-Bench: a wrapper script runs the kit file.
        "Index": {"command": sys.executable, "args": [str(repo / "guard.py"), str(repo / "Index.py")]},
        # A low-level server: its tools come from its live listing.
        "stats": {"command": sys.executable, "args": ["stats_server.py"], "cwd": str(repo)},
    }
    by_server = {
        "Index": _fields("calc_ndvi", "calc_evi", "calc_savi"),
        "stats": _fields("trend_test", "change_point", "mean_series"),
    }
    plan = build_file_plan(by_server, servers, repo)
    assert plan["algorithm"] == "files" and plan["embedding_model"] is None and plan["k"] == 2
    assert {s["slug"]: s["tools"] for s in plan["specialists"]} == {
        "index": ["calc_evi", "calc_ndvi", "calc_savi"],
        "stats_server": ["change_point", "mean_series", "trend_test"],
    }
    assert sorted(plan["files"]) == ["index", "stats_server"]


def test_file_plan_rejects_a_tool_no_file_owns(tmp_path):
    repo = _tool_repo(tmp_path)
    by_server = {"elsewhere": _fields("detect_ships")}
    with pytest.raises(ValueError, match="detect_ships"):
        build_file_plan(by_server, {"elsewhere": {"args": ["/nowhere/server.py"]}}, repo)


def test_real_tool_packages():
    earth = mcp_server_files(task_tool_dir("earth_bench"))
    assert [f.path.stem for f in earth] == ["Analysis", "Index", "Inversion", "Perception", "Statistics"]
    assert sum(len(f.tools) for f in earth) == 104
    for task, server_file in (("geobenchx", "mcp_server.py"), ("eo_gym", "mcp_proxy.py")):
        (found,) = mcp_server_files(task_tool_dir(task))
        assert found.path.name == server_file and found.tools == ()  # tools registered at run time


def test_launched_files_resolve_modules_and_scripts(tmp_path):
    (tmp_path / "srv.py").write_text("")
    module = launched_files({"args": ["-m", "agentic_evals.tasks.geobenchx.tools.mcp_server"]})
    script = launched_files({"args": ["srv.py", "--flag"], "cwd": str(tmp_path)})
    assert [p.name for p in module] == ["mcp_server.py"]
    assert script == [(tmp_path / "srv.py").resolve()]


# A2A.

SAMPLE = contextvars.ContextVar("sample", default="uvicorn")
SKILLS = [
    {
        "id": "vegetation_indices",
        "name": "Vegetation indices",
        "description": "Computes NDVI from red and NIR band paths.",
        "tags": ["ndvi", "raster"],
        "examples": ["Compute NDVI from data/B4.tif and data/B8.tif."],
        "tools": ["calc_ndvi"],
    }
]


class RecordingExecutor(SubAgentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    async def execute(self, context, event_queue):
        self.requests.append((context.task_id, context.context_id))
        await super().execute(context, event_queue)


def test_a2a_card_follows_the_spec_and_runs_in_the_sample_context():
    calls = []

    async def run(text: str, context_id: str, metadata: dict) -> str:
        calls.append((text, context_id, SAMPLE.get()))
        return f"done: {text}"

    async def main():
        SAMPLE.set("sample-1")
        sample_context = contextvars.copy_context()
        SAMPLE.set("other")
        agents = {
            "vegetation": (
                SubAgentExecutor(run, sample_context),
                lambda url: agent_card("Vegetation Index Agent", "Computes vegetation indices.", SKILLS, url),
            )
        }
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            card = await A2ACardResolver(http, urls["vegetation"]).get_agent_card()
            (tool,) = await ask_tools(urls, http)
            first = await tool.ainvoke({"request": "ndvi of a.tif"})
            second = await tool.ainvoke({"request": "and b.tif"})
        return card, tool, first, second

    card, tool, first, second = asyncio.run(main())
    # Required AgentCard fields (spec 4.4.1) and AgentSkill fields (4.4.5).
    assert card.name == "Vegetation Index Agent" and card.description and card.version.startswith("1.0.0+")
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert card.HasField("capabilities") and list(card.default_input_modes) == ["text/plain"]
    assert list(card.default_output_modes) == ["text/plain"]
    (skill,) = card.skills
    assert (skill.id, skill.name, list(skill.tags)) == ("vegetation_indices", "Vegetation indices", ["ndvi", "raster"])
    assert list(skill.examples) == SKILLS[0]["examples"]
    assert "tools" not in {f.name for f, _ in skill.ListFields()}  # our coverage list stays off the card
    # The orchestrator's tool: named by slug, described by the card.
    assert tool.name == "ask_vegetation" and "Vegetation indices: Computes NDVI" in tool.description
    assert 'Example request: "Compute NDVI from data/B4.tif and data/B8.tif."' in tool.description
    assert (first, second) == ("done: ndvi of a.tif", "done: and b.tif")
    assert [c[2] for c in calls] == ["sample-1", "sample-1"]
    assert calls[0][1] and calls[0][1] == calls[1][1]  # one A2A context per specialist


def test_input_required_continues_the_same_task():
    replies = iter(["NEEDS_INPUT: the path of the NIR band", "NDVI = 0.42", "EVI = 0.3"])

    async def run(text: str, context_id: str, metadata: dict) -> str:
        return next(replies)

    executor = RecordingExecutor(run)

    async def main():
        agents = {"veg": (executor, lambda url: agent_card("Veg Agent", "Veg.", SKILLS, url))}
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            (tool,) = await ask_tools(urls, http)
            first = await tool.ainvoke({"request": "NDVI of data/B4.tif"})
            second = await tool.ainvoke({"request": "NIR is data/B8.tif"})
            third = await tool.ainvoke({"request": "Now EVI"})
        return first, second, third

    first, second, third = asyncio.run(main())
    assert first == "[input-required] the path of the NIR band"
    assert (second, third) == ("NDVI = 0.42", "EVI = 0.3")
    (t1, c1), (t2, c2), (t3, c3) = executor.requests
    assert t1 == t2 and c1 == c2  # the follow-up continued the waiting task (spec 3.4.3)
    assert t3 != t1 and c3 == c1  # after completion, a new task in the same context


def test_a2a_failure_is_returned_as_text():
    async def run(text: str, context_id: str, metadata: dict) -> str:
        raise RuntimeError("tool crashed")

    async def main():
        agents = {"x": (SubAgentExecutor(run), lambda url: agent_card("X Agent", "X.", SKILLS, url))}
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            (tool,) = await ask_tools(urls, http)
            return await tool.ainvoke({"request": "go"})

    out = asyncio.run(main())
    assert out.startswith("[failed]") and "tool crashed" in out


def test_shared_trace_numbers_steps_across_agents():
    from uuid import uuid4

    entries: list = []
    a, b = TraceRecorder("a", entries=entries), TraceRecorder("b", entries=entries)

    async def main():
        for recorder, name in ((a, "t1"), (b, "t2"), (a, "t3")):
            run_id = uuid4()
            await recorder.on_tool_start({"name": name}, "{}", run_id=run_id)
            await recorder.on_tool_end("ok", run_id=run_id)

    asyncio.run(main())
    assert [(e["step"], e["agent"], e["name"]) for e in entries] == [(1, "a", "t1"), (2, "b", "t2"), (3, "a", "t3")]


def test_orchestrator_tools_leave_the_specialists():
    from agentic_evals.agents.specialists.clustering import Specialist
    from agentic_evals.agents.specialists.graph import ORCHESTRATOR_PROMPT, orchestrator_prompt, without_tools

    specs = [
        Specialist(slug="a", name="A Agent", description="d", tools=("x", "get_filelist"), skills=(), role_prompt="r"),
        Specialist(slug="b", name="B Agent", description="d", tools=("get_filelist",), skills=(), role_prompt="r"),
    ]
    kept = without_tools(specs, ("get_filelist",))
    assert [s.slug for s in kept] == ["a"]
    assert kept[0].tools == ("x",)
    assert orchestrator_prompt(()) == ORCHESTRATOR_PROMPT
    assert orchestrator_prompt(("get_filelist",)).endswith("Your own tools, besides the ask_ tools: get_filelist.")


def _peer_message(content: str, artifact: dict):
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, artifact=artifact, tool_call_id="call_1", name="ask_veg")


def test_peer_tools_stay_out_of_the_domain_trace():
    from uuid import uuid4

    from agentic_evals.agents.trace_callback import ROLE_METADATA_KEY

    domain_entries: list = []
    domain = TraceRecorder("orchestrator", entries=domain_entries)
    raw = TraceRecorder("orchestrator", roles=None)
    artifact = {"agent": "veg", "task_id": "t1", "context_id": "c1", "state": "completed"}

    async def main():
        for recorder in (domain, raw):
            ask, own = uuid4(), uuid4()
            await recorder.on_tool_start({"name": "ask_veg"}, "{}", run_id=ask, metadata={ROLE_METADATA_KEY: "peer"})
            await recorder.on_tool_end(_peer_message("NDVI = 0.4", artifact), run_id=ask)
            await recorder.on_tool_start({"name": "get_filelist"}, "{}", run_id=own, metadata={})
            await recorder.on_tool_end("a.tif", run_id=own)

    asyncio.run(main())
    assert [(e["name"], e["role"]) for e in domain_entries] == [("get_filelist", "domain")]
    assert [(e["step"], e["name"], e["role"]) for e in raw.entries] == [
        (1, "ask_veg", "peer"),
        (2, "get_filelist", "domain"),
    ]
    assert raw.entries[0]["artifact"] == artifact and raw.entries[0]["output"] == "NDVI = 0.4"
    assert "artifact" not in raw.entries[1]


def test_ask_artifact_names_the_task_the_specialist_calls_belong_to():
    from uuid import uuid4

    entries: list = []
    recorder = TraceRecorder("veg", entries=entries)

    async def run(text: str, context_id: str, metadata: dict) -> str:
        # What langgraph_runner's config metadata does for every tool call of the sub-agent.
        run_id = uuid4()
        await recorder.on_tool_start({"name": "calc_ndvi"}, "{}", run_id=run_id, metadata=metadata)
        await recorder.on_tool_end("0.42", run_id=run_id)
        return "NDVI = 0.42"

    async def main():
        agents = {"veg": (SubAgentExecutor(run), lambda url: agent_card("Veg Agent", "Veg.", SKILLS, url))}
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            (tool,) = await ask_tools(urls, http)
            # As LangGraph's ToolNode calls it: a ToolCall gives a ToolMessage with the artifact.
            message = await tool.ainvoke({"type": "tool_call", "name": tool.name, "args": {"request": "NDVI"}, "id": "c1"})
        return tool, message

    tool, message = asyncio.run(main())
    assert tool.metadata == {"agentic_evals:role": "peer"}
    assert message.content == "NDVI = 0.42"
    assert message.artifact["agent"] == "veg" and message.artifact["state"] == "completed"
    assert message.artifact["context_id"]
    assert [e["a2a_task_id"] for e in entries] == [message.artifact["task_id"]]


def test_subagent_traces_read_each_context_the_orchestrator_used():
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from agentic_evals.agents.specialists.clustering import Specialist
    from agentic_evals.agents.specialists.graph import subagent_traces

    conversation = [
        HumanMessage(content="NDVI of a.tif"),
        AIMessage(content="", tool_calls=[{"name": "calc_ndvi", "args": {"path": "a.tif"}, "id": "x"}]),
        ToolMessage(content="0.42", tool_call_id="x", name="calc_ndvi"),
        AIMessage(content="NDVI = 0.42"),
    ]

    class Graph:
        def __init__(self):
            self.threads = []

        async def aget_state(self, config):
            self.threads.append(config["configurable"]["thread_id"])
            return SimpleNamespace(values={"messages": conversation})

    specs = [
        Specialist(slug="veg", name="Veg Agent", description="d", tools=("calc_ndvi",), skills=(), role_prompt="r"),
        Specialist(slug="ts", name="TS Agent", description="d", tools=("trend",), skills=(), role_prompt="r"),
    ]
    graphs = {"veg": Graph(), "ts": Graph()}
    raw = [
        {"name": "get_filelist", "agent": "orchestrator"},
        {"name": "ask_veg", "artifact": {"agent": "veg", "context_id": "c1", "task_id": "t1"}},
        {"name": "ask_veg", "artifact": {"agent": "veg", "context_id": "c1", "task_id": "t2"}},
    ]
    entries = [{"agent": "orchestrator", "name": "get_filelist"}, {"agent": "veg", "name": "calc_ndvi"}]
    traces = asyncio.run(subagent_traces(specs, graphs, {"veg": "pv", "ts": "pt"}, raw, entries))

    assert graphs["veg"].threads == ["c1"] and graphs["ts"].threads == []
    veg = traces["veg"]
    assert veg["context_ids"] == ["c1"] and veg["system_prompt"] == "pv"
    assert [m["role"] for m in veg["messages"]] == ["human", "ai", "tool", "ai"]
    assert veg["messages"][1]["tool_calls"] == [{"name": "calc_ndvi", "args": {"path": "a.tif"}}]
    assert veg["messages"][2]["name"] == "calc_ndvi"
    assert veg["calls"] == [{"agent": "veg", "name": "calc_ndvi"}]
    assert traces["ts"]["messages"] == [] and traces["ts"]["calls"] == []


def test_an_empty_recorded_trace_is_not_replaced_by_the_orchestrator_messages():
    from types import SimpleNamespace

    from agentic_evals.agents.trace_callback import TRACE_KEY, recorded_trace
    from agentic_evals.tasks.earth_bench.scorers import predicted_calls

    ask = SimpleNamespace(function="ask_veg", arguments={"request": "NDVI"})
    messages = [SimpleNamespace(role="assistant", tool_calls=[ask])]
    assert recorded_trace({}) is None
    assert predicted_calls(SimpleNamespace(store={TRACE_KEY: []}, messages=messages)) == ([], "trace")
    calls, source = predicted_calls(SimpleNamespace(store={}, messages=messages))
    assert source == "messages" and [c["name"] for c in calls] == ["ask_veg"]


def test_each_specialist_request_has_its_own_bridge():
    from inspect_ai.agent import AgentState, agent_bridge
    from inspect_ai.agent._bridge.bridge import _patch_config

    from agentic_evals.agents.specialists.graph import own_bridge

    seen = []

    async def run(text: str, context_id: str, metadata: dict) -> str:
        bridge = _patch_config.get().bridge
        await asyncio.sleep(0.05)  # overlap the two requests
        seen.append((text, bridge, _patch_config.get().bridge is bridge))
        return text

    async def main():
        async with agent_bridge(AgentState(messages=[])) as orchestrator:
            sample_context = contextvars.copy_context()
            agents = {
                name: (SubAgentExecutor(own_bridge(run), sample_context), lambda url, n=name: agent_card(n, "d.", SKILLS, url))
                for name in ("a", "b")
            }
            async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
                tools = await ask_tools(urls, http)
                await asyncio.gather(*(t.ainvoke({"request": t.name}) for t in tools))
            return orchestrator, _patch_config.get().bridge

    orchestrator, after = asyncio.run(main())
    bridges = [bridge for _, bridge, _ in seen]
    assert len(seen) == 2 and all(stable for _, _, stable in seen)  # no request saw the other's bridge
    assert bridges[0] is not bridges[1]
    assert all(bridge is not orchestrator for bridge in bridges)
    assert after is orchestrator and orchestrator.state.messages == []


def test_serve_leaves_no_sdk_task_pending():
    # Left pending, these tasks are destroyed when the sample ends and the SDK's
    # OpenTelemetry spans fail to detach ("created in a different Context").
    async def run(text: str, context_id: str, metadata: dict) -> str:
        return "NEEDS_INPUT: the path"  # an input-required task stays open

    async def main():
        agents = {"veg": (SubAgentExecutor(run), lambda url: agent_card("Veg Agent", "Veg.", SKILLS, url))}
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            (tool,) = await ask_tools(urls, http)
            await tool.ainvoke({"request": "NDVI"})
        await asyncio.sleep(0.1)
        me = asyncio.current_task()
        return [t.get_coro().__qualname__ for t in asyncio.all_tasks() if t is not me and not t.done()]

    assert asyncio.run(main()) == []


def test_a_request_names_the_tool_run_that_sent_it():
    seen = {}
    raw = TraceRecorder("orchestrator", roles=None)

    async def run(text: str, context_id: str, metadata: dict) -> str:
        seen.update(metadata)
        return "ok"

    async def main():
        agents = {"veg": (SubAgentExecutor(run), lambda url: agent_card("Veg Agent", "Veg.", SKILLS, url))}
        async with serve(agents) as urls, httpx.AsyncClient(timeout=30) as http:
            (tool,) = await ask_tools(urls, http)
            call = {"type": "tool_call", "name": tool.name, "args": {"request": "NDVI"}, "id": "c1"}
            await tool.ainvoke(call, config={"callbacks": [raw]})

    asyncio.run(main())
    assert seen["caller_run_id"] == raw.entries[0]["run_id"]
    assert seen["a2a_task_id"]


def test_weave_recorder_nests_a_specialist_under_the_ask_call():
    from types import SimpleNamespace
    from uuid import uuid4

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from agentic_evals.agents.trace_callback import ROLE_METADATA_KEY
    from agentic_evals.agents.weave_trace import WeaveRecorder

    class Client:
        def __init__(self):
            self.calls = []

        def create_call(self, op, inputs, parent=None, attributes=None, display_name=None, use_stack=True):
            call = SimpleNamespace(id=str(uuid4()), op=op, name=display_name, parent=parent, attributes=attributes or {}, output=None)
            self.calls.append(call)
            return call

        def finish_call(self, call, output=None, exception=None):
            call.output = output if exception is None else repr(exception)

    client, solver = Client(), SimpleNamespace(id="solver")
    weave = WeaveRecorder(client, solver)
    orch, ask, spec, model, tool = (uuid4() for _ in range(5))

    def reply(text):
        return LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]])

    async def main():
        await weave.on_chain_start({}, {"messages": [HumanMessage("Q?")]}, run_id=orch, name="orchestrator")
        await weave.on_tool_start({"name": "ask_veg"}, "", run_id=ask, parent_run_id=orch, metadata={ROLE_METADATA_KEY: "peer"})
        # The specialist's own graph run, started over A2A while ask_veg is open.
        await weave.on_chain_start({}, {"messages": [HumanMessage("NDVI")]}, run_id=spec, name="veg", metadata={"caller_run_id": str(ask)})
        await weave.on_chat_model_start({}, [[HumanMessage("NDVI")]], run_id=model, parent_run_id=spec)
        await weave.on_llm_end(reply("calling"), run_id=model)
        await weave.on_tool_start({"name": "calc_ndvi"}, "", run_id=tool, parent_run_id=spec, inputs={"path": "a.tif"})
        await weave.on_tool_end("0.42", run_id=tool)
        await weave.on_chain_end({"messages": [AIMessage("NDVI = 0.42")]}, run_id=spec)
        await weave.on_tool_end(_peer_message("NDVI = 0.42", {"task_id": "t1"}), run_id=ask)
        await weave.on_chain_end({"messages": [AIMessage("done")]}, run_id=orch)

    asyncio.run(main())
    by_name = {c.name: c for c in client.calls}
    assert by_name["orchestrator"].parent is solver
    assert by_name["ask_veg"].parent is by_name["orchestrator"]
    assert by_name["veg"].parent is by_name["ask_veg"]
    assert by_name["veg · model"].parent is by_name["veg"] and by_name["calc_ndvi"].parent is by_name["veg"]
    assert by_name["ask_veg"].output["a2a"] == {"task_id": "t1"} and by_name["ask_veg"].attributes["role"] == "peer"
    assert by_name["calc_ndvi"].output == {"output": "0.42", "error": False}
    assert by_name["veg"].output == {"answer": "NDVI = 0.42"}
    assert all(c.output is not None for c in client.calls)
