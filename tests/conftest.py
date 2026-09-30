from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_evals.paths import EARTH_AGENT_ROOT


def _load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def earth_agent_root() -> Path:
    if not (EARTH_AGENT_ROOT / "extracted_tool_calls_GT.json").exists():
        pytest.skip("Earth-Agent checkout missing; run `uv run python scripts/fetch_upstream.py earth-agent --skip-venv`")
    return EARTH_AGENT_ROOT


@pytest.fixture(scope="session")
def earth_bench_gold(earth_agent_root: Path) -> list[dict]:
    return _load(earth_agent_root / "extracted_tool_calls_GT.json")


@pytest.fixture(scope="session")
def earth_bench_questions(earth_agent_root: Path) -> dict:
    return _load(earth_agent_root / "benchmark" / "question.json")


def published_run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "evaluate_langchain").iterdir() if (p / "extracted_tool_calls.json").exists())
