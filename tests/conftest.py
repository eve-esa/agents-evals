from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from agentic_evals.tasks.earth_bench.dataset import TASK_DATA

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PUBLISHED_RUNS = FIXTURES / "earth_bench" / "published_runs"


def read_json(path: Path):
    """Read a JSON file, or ``path`` with ``.gz`` appended."""
    gzipped = Path(str(path) + ".gz")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if gzipped.is_file():
        with gzip.open(gzipped, "rt", encoding="utf-8") as handle:
            return json.loads(handle.read())
    raise FileNotFoundError(path)


@pytest.fixture(scope="session")
def earth_agent_root() -> Path:
    """Earth-Bench task files. The name is kept for the tests that grew up around the checkout."""
    return TASK_DATA


@pytest.fixture(scope="session")
def earth_bench_gold(earth_agent_root: Path) -> list[dict]:
    return read_json(earth_agent_root / "extracted_tool_calls_GT.json")


@pytest.fixture(scope="session")
def earth_bench_questions(earth_agent_root: Path) -> dict:
    return read_json(earth_agent_root / "question.json")


def published_run_dirs(root: Path | None = None) -> list[Path]:
    del root
    return sorted(p for p in PUBLISHED_RUNS.iterdir() if (p / "extracted_tool_calls.json.gz").is_file())
