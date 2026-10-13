"""Split a task's tools into specialists, offline, and commit the split.

Three algorithms, chosen per benchmark in ``embedding.yaml`` (``--algorithm`` overrides):

- ``ward`` / ``kmeans`` (Tool-Planner, Liu et al., ICLR 2025, arXiv 2406.03807):
  embed each tool's name, description and argument names, then cluster the
  vectors. Tool-Planner used k-means++ and did best on ToolBench with about nine
  tools per cluster, so that is the default size. Ward agglomerative clustering
  on L2-normalised vectors approximates cosine distance, is deterministic, and
  keeps clusters balanced. Each cluster gets a provisional label from its most
  distinctive terms by class-based TF-IDF (BERTopic).
- ``files``: one specialist per source file in the task's tool package that
  defines an MCP server (``FastMCP(...)`` or ``Server(...)`` from ``fastmcp`` /
  ``mcp.server``). A file owns the functions decorated ``@<server>.tool``. A
  server that registers its tools at run time (``@server.list_tools()``) owns
  the tools the server config launching that file lists. No embedding call.

The describer (``describer``) then writes each specialist's A2A AgentCard text,
its AgentSkills and its role prompt. The result is committed under
``clusters/<task>.json``::

    uv run python -m agentic_evals.agents.specialists.clustering --task earth_bench

The file maps each tool to a digest of its name and description, and lists the
specialists with the tools each owns. An eval only reads these files: it uses
the first one that covers the sample's tools (every tool present with the same
digest), so a sample that exposes a subset of a benchmark's tools (EO-Gym) gets
the same split restricted to that subset. The embeddings and chat APIs are
called only by this CLI, or by the solver with ``recompute=True`` when no file
covers the tools.

Endpoints come from ``embedding.yaml`` next to this file (any OpenAI-compatible
server; OpenRouter by default).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from langchain_core.tools import BaseTool

from agentic_evals.agents.specialists.describer import (
    ToolFields,
    describe_specialists,
    load_describer_config,
    placeholder_specialist,
    tool_skill,
)
from agentic_evals.paths import REPO_ROOT

CLUSTERS_DIR = Path(__file__).resolve().parent / "clusters"
EMBEDDING_CONFIG = Path(__file__).resolve().parent / "embedding.yaml"
TOOLS_PER_AGENT = 9
MAX_AGENTS = 12
ALGORITHMS = ("ward", "kmeans", "files")
_NAME = re.compile(r"[^A-Za-z0-9_-]+")
# Words every tool docstring and argument list shares; they say nothing about a cluster.
_SCHEMA_WORDS = frozenset(
    ["arg", "args", "bool", "default", "dict", "dir", "file", "files", "float", "input", "int", "list", "optional", "output", "path", "paths", "result", "return", "returns", "save", "str", "temp", "tif", "value", "values"]
)


@dataclass(frozen=True)
class Specialist:
    slug: str
    name: str
    description: str
    tools: tuple[str, ...]
    skills: tuple[dict[str, Any], ...] = ()
    role_prompt: str = ""


def tool_text(name: str, description: str, args: Sequence[str] = ()) -> str:
    text = f"{name}: {description.strip()}"
    return f"{text} Args: {', '.join(args)}." if args else text


def tool_digest(name: str, description: str) -> str:
    return hashlib.sha256(f"{name}\n{description.strip()}".encode()).hexdigest()[:16]


def _tool_fields(tool: BaseTool) -> ToolFields:
    return tool.name, tool.description or "", list(tool.args)


@dataclass(frozen=True)
class EmbeddingConfig:
    """The embedding endpoint and per-benchmark clustering settings, from ``embedding.yaml``.

    The key itself stays in the environment.
    """

    base_url: str
    model: str
    api_key_env: str
    dimensions: int | None = None
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set; put it in .env (named by api_key_env in embedding.yaml)")
        return key


def load_embedding_config(path: Path = EMBEDDING_CONFIG) -> EmbeddingConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [f for f in ("base_url", "model", "api_key_env") if not data.get(f)]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return EmbeddingConfig(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        api_key_env=str(data["api_key_env"]),
        dimensions=data.get("dimensions"),
        benchmarks=dict(data.get("benchmarks") or {}),
    )


_SETTING_KEYS = {"algorithm", "n_agents", "tools_per_agent", "max_agents"}


def benchmark_settings(task: str, config: EmbeddingConfig) -> dict[str, Any]:
    """Clustering settings for ``task``: the defaults, overridden by its ``benchmarks`` entry."""
    settings: dict[str, Any] = {
        "algorithm": "ward",
        "n_agents": None,
        "tools_per_agent": TOOLS_PER_AGENT,
        "max_agents": MAX_AGENTS,
    }
    override = dict(config.benchmarks.get(task) or {})
    unknown = set(override) - _SETTING_KEYS
    if unknown:
        raise ValueError(f"benchmarks.{task}: unknown keys {', '.join(sorted(unknown))}")
    settings.update(override)
    if settings["algorithm"] not in ALGORITHMS:
        raise ValueError(f"benchmarks.{task}.algorithm must be one of {', '.join(ALGORITHMS)}, got {settings['algorithm']!r}")
    return settings


async def embed(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    """L2-normalised embeddings from the endpoint in ``config``."""
    from langchain_openai import OpenAIEmbeddings

    client = OpenAIEmbeddings(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key(),
        dimensions=config.dimensions,
        # Sends the raw strings; tiktoken pre-tokenisation only fits OpenAI's own models.
        check_embedding_ctx_length=False,
    )
    vectors = np.asarray(await client.aembed_documents(texts), dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def cluster(vectors: np.ndarray, k: int, algorithm: str = "ward") -> np.ndarray:
    """Cluster label per row."""
    if k <= 1:
        return np.zeros(len(vectors), dtype=int)
    if algorithm == "ward":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(vectors)
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit_predict(vectors)
    raise ValueError(f"algorithm must be 'ward' or 'kmeans', got {algorithm!r}")


def choose_k(
    vectors: np.ndarray,
    n_agents: int | str | None = None,
    tools_per_agent: int = TOOLS_PER_AGENT,
    max_agents: int = MAX_AGENTS,
    algorithm: str = "ward",
) -> int:
    """``n_agents`` if given, the best cosine silhouette for ``"auto"``, else ``n / tools_per_agent``."""
    n = len(vectors)
    if n_agents == "auto":
        from sklearn.metrics import silhouette_score

        candidates = range(2, min(max_agents, n - 1) + 1)
        if not candidates:
            return 1
        return max(
            candidates,
            key=lambda k: silhouette_score(vectors, cluster(vectors, k, algorithm), metric="cosine"),
        )
    k = int(n_agents) if n_agents else round(n / tools_per_agent)
    return max(1, min(k, max_agents, n))


def label_clusters(texts: list[str], labels: np.ndarray, top: int = 4) -> list[list[str]]:
    """Most distinctive terms of each cluster, by class-based TF-IDF (BERTopic's c-TF-IDF)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    k = int(labels.max()) + 1
    docs = [" ".join(t for t, label in zip(texts, labels) if label == c).replace("_", " ") for c in range(k)]
    vectorizer = CountVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | _SCHEMA_WORDS), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    counts = vectorizer.fit_transform(docs).toarray().astype(float)
    terms = vectorizer.get_feature_names_out()
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    idf = np.log(1.0 + counts.sum(axis=1).mean() / np.maximum(counts.sum(axis=0), 1.0))
    scores = tf * idf
    return [[str(terms[i]) for i in np.argsort(-row)[:top] if row[i] > 0] for row in scores]


def _digests(tools: list[ToolFields]) -> dict[str, str]:
    return {name: tool_digest(name, description) for name, description, _ in tools}


async def build_plan(
    tools: list[ToolFields],
    *,
    embedding: EmbeddingConfig | None = None,
    n_agents: int | str | None = None,
    tools_per_agent: int = TOOLS_PER_AGENT,
    max_agents: int = MAX_AGENTS,
    algorithm: str = "ward",
    embedder: Callable[[list[str], EmbeddingConfig], Any] | None = None,
) -> dict[str, Any]:
    """Cluster ``(name, description, arg names)`` tools into the JSON written to ``clusters/``.

    ``embedding`` defaults to ``embedding.yaml``. ``embedder(texts, config)`` returns
    normalised vectors (default ``embed``). Specialists carry placeholder text
    until ``describe_specialists`` fills it.
    """
    embedding = embedding or load_embedding_config()
    tools = sorted(tools)
    texts = [tool_text(*t) for t in tools]
    vectors = np.asarray(await (embedder or embed)(texts, embedding), dtype=float)
    k = choose_k(vectors, n_agents, tools_per_agent, max_agents, algorithm)
    labels = cluster(vectors, k, algorithm)
    # Number clusters by their first tool name, so the order does not depend on the algorithm's ids.
    order = sorted(range(k), key=lambda c: next(i for i, label in enumerate(labels) if label == c))
    labels = np.asarray([order.index(int(label)) for label in labels])
    terms = label_clusters(texts, labels)
    specialists = []
    for c in range(k):
        label = _NAME.sub("", "_".join([str(c + 1), *terms[c][:2]]))[:40]
        specialists.append(placeholder_specialist(label, [t for t, lab in zip(tools, labels) if lab == c]))
    return {
        "embedding_model": embedding.model,
        "algorithm": algorithm,
        "k": k,
        "tools": _digests(tools),
        "specialists": specialists,
    }


# ``files``: which source files define an MCP server, and which tools each one owns.

_SERVER_CLASSES = {"FastMCP", "Server"}
_SERVER_MODULE_PREFIXES = ("fastmcp", "mcp.server")


@dataclass(frozen=True)
class ServerFile:
    path: Path
    server_vars: tuple[str, ...]
    tools: tuple[str, ...]
    """Tools decorated ``@<server>.tool``; empty when the server registers them at run time."""


def _is_server_module(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _SERVER_MODULE_PREFIXES)


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _constructors(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """Names bound to an MCP server class, and module aliases (``import mcp.server as s``)."""
    classes: set[str] = set()
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and _is_server_module(node.module):
            classes.update(a.asname or a.name for a in node.names if a.name in _SERVER_CLASSES)
        elif isinstance(node, ast.Import):
            for a in node.names:
                modules[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
    return classes, modules


def _makes_server(call: ast.expr, classes: set[str], modules: dict[str, str]) -> bool:
    if not isinstance(call, ast.Call):
        return False
    dotted = _dotted(call.func)
    if dotted is None:
        return False
    if dotted in classes:
        return True
    head, _, rest = dotted.partition(".")
    module, _, cls = f"{modules.get(head, head)}.{rest}".rpartition(".")
    return bool(rest) and head in modules and cls in _SERVER_CLASSES and _is_server_module(module)


def _decorated_tools(tree: ast.Module, server_vars: set[str]) -> list[str]:
    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Attribute) and target.attr == "tool" and _dotted(target.value) in server_vars:
                name = node.name
                if isinstance(deco, ast.Call):
                    for kw in deco.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            name = kw.value.value
                tools.append(name)
    return tools


def mcp_server_files(tool_dir: Path) -> list[ServerFile]:
    """The ``*.py`` files in ``tool_dir`` that construct an MCP server, with their decorated tools."""
    found = []
    for path in sorted(tool_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes, modules = _constructors(tree)
        server_vars: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _makes_server(node.value, classes, modules):
                server_vars.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
                and _makes_server(node.value, classes, modules)
            ):
                server_vars.add(node.target.id)
        if server_vars:
            found.append(ServerFile(path.resolve(), tuple(sorted(server_vars)), tuple(_decorated_tools(tree, server_vars))))
    return found


def launched_files(config: dict[str, Any]) -> list[Path]:
    """Source files a server config runs: script arguments, and ``-m module`` resolved to its file."""
    cwd = Path(config.get("cwd") or ".")
    args = [str(a) for a in config.get("args") or []]
    paths = []
    for i, arg in enumerate(args):
        if arg == "-m" and i + 1 < len(args):
            spec = importlib.util.find_spec(args[i + 1])
            if spec is not None and spec.origin:
                paths.append(Path(spec.origin).resolve())
        elif arg.endswith(".py"):
            path = Path(arg) if Path(arg).is_absolute() else cwd / arg
            if path.exists():
                paths.append(path.resolve())
    return paths


def build_file_plan(
    fields_by_server: dict[str, list[ToolFields]],
    servers: dict[str, dict[str, Any]],
    tool_dir: Path,
) -> dict[str, Any]:
    """One specialist per MCP server file in ``tool_dir``; see the module docstring."""
    candidates = mcp_server_files(tool_dir)
    if not candidates:
        raise ValueError(f"no file in {tool_dir} defines an MCP server")
    by_path = {c.path: c for c in candidates}
    listed = {fields[0]: fields for server_fields in fields_by_server.values() for fields in server_fields}
    owner: dict[str, Path] = {}
    for c in candidates:
        for tool in c.tools:
            if tool in listed:
                owner.setdefault(tool, c.path)
    for server, fields in fields_by_server.items():
        files = [p for p in launched_files(servers.get(server, {})) if p in by_path]
        for name, _, _ in fields:
            if name not in owner and files:
                owner[name] = files[0]
    unassigned = sorted(set(listed) - set(owner))
    if unassigned:
        raise ValueError(f"tools that no MCP server file in {tool_dir} defines or launches: {', '.join(unassigned)}")
    specialists, files = [], {}
    for c in candidates:
        owned = sorted(name for name, path in owner.items() if path == c.path)
        if not owned:
            print(f"note: {c.path.name} defines an MCP server but no listed tool belongs to it; skipped")
            continue
        label = c.path.stem.lower()
        specialists.append(placeholder_specialist(label, [listed[name] for name in owned]))
        files[label] = str(c.path.relative_to(REPO_ROOT)) if c.path.is_relative_to(REPO_ROOT) else str(c.path)
    return {
        "embedding_model": None,
        "algorithm": "files",
        "k": len(specialists),
        "files": files,
        "tools": _digests(list(listed.values())),
        "specialists": specialists,
    }


def _slug(spec: dict[str, Any]) -> str:
    # Plans written before slugs existed name specialists by their label only.
    return spec.get("slug") or spec["name"]


def _check(plan: dict[str, Any], source: str) -> dict[str, Any]:
    # ``orchestrator`` lists tools the orchestrator runs itself (graph.orchestrator_tools).
    owned = [t for s in plan["specialists"] for t in s["tools"]] + plan.get("orchestrator", [])
    if sorted(owned) != sorted(plan["tools"]):
        raise ValueError(f"{source}: every tool must belong to exactly one specialist or to the orchestrator")
    slugs = [_slug(s) for s in plan["specialists"]]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"{source}: specialist slugs must be unique")
    return plan


@cache
def _load(path: str) -> dict[str, Any]:
    return _check(json.loads(Path(path).read_text(encoding="utf-8")), path)


def _covers(plan: dict[str, Any], digests: dict[str, str]) -> bool:
    return all(plan["tools"].get(name) == digest for name, digest in digests.items())


def _restrict(plan: dict[str, Any], names: set[str], descriptions: dict[str, str] | None = None) -> list[Specialist]:
    """The plan's specialists cut down to ``names``: absent tools leave skills, and empty skills go."""
    specialists = []
    for s in plan["specialists"]:
        owned = tuple(t for t in s["tools"] if t in names)
        if not owned:
            continue
        slug = _slug(s)
        raw_skills = s.get("skills") or [tool_skill(t, (descriptions or {}).get(t, ""), slug) for t in s["tools"]]
        skills = []
        for skill in raw_skills:
            tools = [t for t in skill.get("tools", owned) if t in names]
            if tools:
                skills.append({**skill, "tools": tools})
        specialists.append(
            Specialist(slug, s["name"], s["description"], owned, tuple(skills), s.get("role_prompt", ""))
        )
    return specialists


def find_plan(digests: dict[str, str], clusters: str = "auto", directory: Path = CLUSTERS_DIR) -> dict[str, Any] | None:
    """The committed plan that covers these tools: ``clusters`` is a path, or ``"auto"`` to search ``directory``."""
    paths = sorted(directory.glob("*.json")) if clusters == "auto" else [Path(clusters)]
    for path in paths:
        plan = _load(str(path.resolve()))
        if _covers(plan, digests):
            return plan
    return None


_recompute_lock: asyncio.Lock | None = None


async def plan_specialists(
    tools: list[BaseTool],
    clusters: str = "auto",
    recompute: bool = False,
    directory: Path = CLUSTERS_DIR,
) -> list[Specialist]:
    """The specialists for these tools, from the committed clusters.

    Raises ``LookupError`` when no committed file covers the tools, unless
    ``recompute``: then the tools are clustered (``ward``, default size) and
    described once, and the plan is written to ``directory`` for every later
    sample and run.
    """
    global _recompute_lock
    fields = [_tool_fields(t) for t in tools]
    digests = _digests(fields)
    plan = find_plan(digests, clusters, directory)
    if plan is None and recompute:
        _recompute_lock = _recompute_lock or asyncio.Lock()
        async with _recompute_lock:
            plan = find_plan(digests, clusters, directory)
            if plan is None:
                plan = await describe_specialists(
                    await build_plan(fields), fields, load_describer_config(EMBEDDING_CONFIG)
                )
                fingerprint = hashlib.sha256(json.dumps(sorted(digests.items())).encode()).hexdigest()[:12]
                write_plan(plan, directory / f"auto_{fingerprint}.json")
    if plan is None:
        raise LookupError(
            f"No tool clusters in {directory if clusters == 'auto' else clusters} cover these "
            f"{len(tools)} tools. Run `uv run python -m agentic_evals.agents.specialists.clustering "
            "--task <task>` and commit the file, or pass recompute=True."
        )
    return _restrict(plan, set(digests), {name: description for name, description, _ in fields})


def write_plan(plan: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_check(plan, str(path)), indent=2) + "\n", encoding="utf-8")
    _load.cache_clear()
    return path


TASKS = ("earth_bench", "geobenchx", "eo_gym")


def task_tool_dir(task: str) -> Path:
    """The task's tool package, where ``files`` looks for MCP server files."""
    return Path(importlib.import_module(f"agentic_evals.tasks.{task}.tools").__file__).resolve().parent


def _task_servers(task: str) -> dict[str, dict[str, Any]]:
    """All of a benchmark's MCP servers, configured only to list tools."""
    if task == "earth_bench":
        from agentic_evals.tasks.earth_bench.tools import kit_servers

        return kit_servers(Path(tempfile.mkdtemp(prefix="specialists-")))
    if task == "geobenchx":
        from unittest import mock

        from agentic_evals.tasks.geobenchx import tools as geobenchx_tools

        try:
            return geobenchx_tools.sample_servers(None)
        except FileNotFoundError:
            # Listing tools does not read the datasets, so empty directories do.
            empty = Path(tempfile.mkdtemp(prefix="specialists-"))
            dirs = (empty / "StatData", empty / "Geodata")
            for path in dirs:
                path.mkdir()
            with (
                mock.patch.object(geobenchx_tools, "data_directories", lambda: dirs),
                mock.patch.object(geobenchx_tools, "GEOBENCHX_DATA", empty),
            ):
                return geobenchx_tools.sample_servers(None)
    if task == "eo_gym":
        from agentic_evals.tasks.eo_gym.dataset import TOOLS_JSON
        from agentic_evals.tasks.eo_gym.tools import proxy_servers

        all_tools = json.loads(Path(TOOLS_JSON).read_text())["all_tools"]
        # Listing tools never reaches /execute, so the tool server need not run.
        return proxy_servers(all_tools, "", "http://127.0.0.1:9/execute")
    raise ValueError(f"unknown task {task!r}; expected {', '.join(TASKS)}")


async def _cluster_task(args: argparse.Namespace) -> Path:
    from agentic_evals.agents.langgraph_react import load_mcp_tools_from

    servers = _task_servers(args.task)
    fields_by_server: dict[str, list[ToolFields]] = {}
    async with AsyncExitStack() as stack:
        for server, server_config in servers.items():
            tools = await load_mcp_tools_from({server: server_config}, stack)
            fields_by_server[server] = [_tool_fields(t) for t in tools]
    fields = [f for server_fields in fields_by_server.values() for f in server_fields]
    config = load_embedding_config(Path(args.config))
    settings = benchmark_settings(args.task, config)
    for key in _SETTING_KEYS:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if settings["algorithm"] == "files":
        plan = build_file_plan(fields_by_server, servers, task_tool_dir(args.task))
    else:
        n_agents = settings["n_agents"]
        if n_agents not in (None, "auto"):
            n_agents = int(n_agents)
        plan = await build_plan(
            fields,
            embedding=config,
            n_agents=n_agents,
            tools_per_agent=int(settings["tools_per_agent"]),
            max_agents=int(settings["max_agents"]),
            algorithm=settings["algorithm"],
        )
    if not args.no_describe:
        plan = await describe_specialists(plan, fields, load_describer_config(Path(args.config)))
    path = write_plan(plan, Path(args.out) if args.out else CLUSTERS_DIR / f"{args.task}.json")
    print(f"{len(fields)} tools -> {plan['k']} specialists ({settings['algorithm']}); wrote {path}")
    for s in plan["specialists"]:
        skills = ", ".join(skill["id"] for skill in s["skills"])
        print(f"  {s['slug']} = {s['name']} ({len(s['tools'])} tools; skills: {skills})")
    return path


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Split a benchmark's tools into specialists.")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--config", default=str(EMBEDDING_CONFIG), help="endpoints and per-benchmark settings YAML")
    # Unset flags fall back to the task's entry under ``benchmarks`` in the YAML.
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=None)
    parser.add_argument("--n-agents", default=None, help="number of specialists, or 'auto' (silhouette)")
    parser.add_argument("--tools-per-agent", type=int, default=None)
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--no-describe", action="store_true", help="placeholder A2A text, no chat calls")
    parser.add_argument("--out", default=None, help=f"output path (default {CLUSTERS_DIR}/<task>.json)")
    asyncio.run(_cluster_task(parser.parse_args()))


if __name__ == "__main__":
    main()
