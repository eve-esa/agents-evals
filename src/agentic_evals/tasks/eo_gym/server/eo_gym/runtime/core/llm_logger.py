"""
Optional JSONL logger for LLM interactions.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "llm_interactions.jsonl"
LOG_ENABLED_ENV = "EO_GYM_LLM_LOG_ENABLED"
LOG_PATH_ENV = "EO_GYM_LLM_LOG_PATH"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def is_llm_logging_enabled() -> bool:
    return _env_flag(LOG_ENABLED_ENV, False)


def get_llm_log_path() -> Path:
    raw = os.getenv(LOG_PATH_ENV)
    if not raw:
        return DEFAULT_LOG_PATH
    return Path(raw).expanduser()


def log_interaction(
    *,
    source: str,
    interaction_type: str,
    input_payload: Any,
    response: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a structured record to the shared log when LLM logging is enabled.
    """
    if not is_llm_logging_enabled():
        return

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": source,
        "type": interaction_type,
        "input": input_payload,
        "response": response,
    }
    if metadata:
        entry["metadata"] = metadata

    log_path = get_llm_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
