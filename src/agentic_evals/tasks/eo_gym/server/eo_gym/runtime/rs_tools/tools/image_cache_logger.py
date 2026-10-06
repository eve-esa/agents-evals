"""
Structured append-only logger for image cache transformations (crop/mask).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import eo_gym.runtime.core.config_loader as config_loader


def _resolve_log_path() -> Path:
    override = os.getenv("IMAGE_CACHE_LOG_PATH")
    if override:
        return Path(override)
    return config_loader.IMAGE_CACHE_DIR / "image_cache_ops.log"


LOG_PATH = _resolve_log_path()


def log_image_event(event: str, payload: Dict[str, Any]) -> None:
    """
    Append a single structured event to the image cache log.

    The logger is intentionally resilient: failures are printed but do not raise.
    """
    try:
        log_path = _resolve_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"event": event, "timestamp": time.time(), **(payload or {})}
        fh = None
        try:
            fh = log_path.open("a+", encoding="utf-8")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fh is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    fh.close()
                except Exception:
                    pass
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[WARN] Failed to write image cache log: {exc}")


__all__ = ["log_image_event", "LOG_PATH"]
