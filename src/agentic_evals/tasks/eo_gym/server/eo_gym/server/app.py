from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from eo_gym.runtime.core import config_loader


class ToolRequest(BaseModel):
    tool_name: Optional[str] = None
    tool: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    params: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None


class ResponseCache:
    """Thread-safe in-memory TTL + LRU cache for tool execution responses."""

    def __init__(self, enabled: bool, max_entries: int, ttl_secs: float, cache_errors: bool):
        self.enabled = bool(enabled)
        self.max_entries = max(1, int(max_entries))
        self.ttl_secs = max(0.0, float(ttl_secs))
        self.cache_errors = bool(cache_errors)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _expired(self, created_at: float, now: float) -> bool:
        return self.ttl_secs > 0.0 and (now - created_at) > self.ttl_secs

    def _cleanup_expired_locked(self, now: float) -> None:
        if self.ttl_secs <= 0.0:
            return
        expired_keys: List[str] = []
        for key, (created_at, _) in self._entries.items():
            if self._expired(created_at, now):
                expired_keys.append(key)
        for key in expired_keys:
            self._entries.pop(key, None)

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired_locked(now)
            item = self._entries.get(key)
            if item is None:
                self.misses += 1
                return None
            created_at, value = item
            if self._expired(created_at, now):
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: str, is_error: bool = False) -> None:
        if not self.enabled:
            return
        if is_error and not self.cache_errors:
            return
        now = time.monotonic()
        with self._lock:
            self._cleanup_expired_locked(now)
            self._entries[key] = (now, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._cleanup_expired_locked(now)
            return {
                "enabled": self.enabled,
                "max_entries": self.max_entries,
                "ttl_secs": self.ttl_secs,
                "cache_errors": self.cache_errors,
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class ExecuteLimiter:
    """Thread-safe limiter for concurrent tool execution inside `/execute`."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max(1, int(max_concurrent))
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self._active = 0
        self._peak_active = 0
        self.total_calls = 0
        self.total_wait_secs = 0.0

    def run(self, fn: Callable[[], Any]) -> Any:
        wait_started = time.monotonic()
        self._semaphore.acquire()
        wait_secs = time.monotonic() - wait_started
        with self._lock:
            self.total_wait_secs += wait_secs
            self.total_calls += 1
            self._active += 1
            if self._active > self._peak_active:
                self._peak_active = self._active
        try:
            return fn()
        finally:
            with self._lock:
                if self._active > 0:
                    self._active -= 1
            self._semaphore.release()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_wait = self.total_wait_secs / self.total_calls if self.total_calls else 0.0
            return {
                "max_concurrent": self.max_concurrent,
                "active": self._active,
                "peak_active": self._peak_active,
                "total_calls": self.total_calls,
                "total_wait_secs": self.total_wait_secs,
                "avg_wait_secs": avg_wait,
            }


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _parse_preload_list(raw: str | Sequence[str] | None) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if len(values) == 1 and values[0].lower() == "none":
        return []
    return values


def _normalize_arguments_for_cache(arguments: Dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(arguments)


def _build_cache_key(tool_name: str, arguments: Dict[str, Any], image_path: str) -> str:
    arg_key = _normalize_arguments_for_cache(arguments)
    return f"{tool_name}\n{image_path}\n{arg_key}"


def build_executor(device: str):
    from eo_gym.runtime.core.shared_tool_executor import ToolExecutor

    return ToolExecutor(device=device, log_calls=False)


def preload_executor_tools(executor: Any, tool_names: Sequence[str]) -> None:
    if not tool_names:
        return

    load_tool = getattr(executor, "_load_tool", None)
    if not callable(load_tool):
        print("Warning: executor does not support tool preloading; skipping.")
        return

    for tool_name in tool_names:
        try:
            load_tool(tool_name)
        except Exception as exc:
            print(f"Warning: failed to preload tool '{tool_name}': {exc}")


def preload_models(device: str, model_names: Sequence[str]) -> None:
    if not model_names:
        return

    alias_map = {
        "qwen": "ai_optical_analysis",
        "local_vlm": "ai_optical_analysis",
        "ai_optical": "ai_optical_analysis",
        "ai_optical_analysis_qwen": "ai_optical_analysis",
        "owlv2": "optical_sar",
        "optical_sar_bbox": "optical_sar",
        "get_object_bbox_optical_sar": "optical_sar",
    }
    selected_raw = {name.strip().lower() for name in model_names if name.strip()}
    normalized_raw = {alias_map.get(name, name) for name in selected_raw}

    supported_models = ("sam3", "dino", "ai_optical_analysis", "optical_sar")
    selected = {name for name in normalized_raw if name in supported_models}
    if "all" in normalized_raw or "*" in normalized_raw:
        selected.update(supported_models)

    if not selected:
        unknown = sorted(name for name in normalized_raw if name not in {"none", "all", "*"})
        if unknown:
            print(f"Warning: unsupported preload model names ignored: {unknown}")
        return

    print(f"Preloading EO model caches: {','.join(sorted(selected))}")

    if "sam3" in selected or "dino" in selected:
        try:
            from eo_gym.runtime.rs_tools.tools.model_cache import (
                preload_grounding_dino_components,
                preload_sam3_components,
            )
        except Exception as exc:
            print(f"Warning: failed to import SAM3/DINO preload helpers: {exc}")
            preload_grounding_dino_components = None
            preload_sam3_components = None

        if "sam3" in selected and callable(preload_sam3_components):
            try:
                preload_sam3_components(device=device)
            except Exception as exc:
                print(f"Warning: failed to preload SAM3: {exc}")
        if "dino" in selected and callable(preload_grounding_dino_components):
            try:
                preload_grounding_dino_components(device=device)
            except Exception as exc:
                print(f"Warning: failed to preload Grounding DINO: {exc}")

    if "ai_optical_analysis" in selected:
        try:
            from eo_gym.runtime.rs_tools.tools.ai_optical_analysis import preload_qwen_components
        except Exception as exc:
            print(f"Warning: failed to import ai_optical_analysis preload helper: {exc}")
            preload_qwen_components = None
        if callable(preload_qwen_components):
            try:
                preload_qwen_components()
            except Exception as exc:
                print(f"Warning: failed to preload ai_optical_analysis local VLM: {exc}")

    if "optical_sar" in selected:
        try:
            from eo_gym.runtime.rs_tools.tools.get_object_bbox_optical_sar import preload_optical_sar_detector_components
        except Exception as exc:
            print(f"Warning: failed to import optical_sar preload helper: {exc}")
            preload_optical_sar_detector_components = None
        if callable(preload_optical_sar_detector_components):
            try:
                preload_optical_sar_detector_components(device=device)
            except Exception as exc:
                print(f"Warning: failed to preload optical_sar detector: {exc}")


def create_app(
    executor: Any,
    response_cache: Optional[ResponseCache] = None,
    max_concurrent_executes: int = 1,
) -> FastAPI:
    app = FastAPI(title="EO Tool API Server")
    execute_limiter = ExecuteLimiter(max_concurrent=max_concurrent_executes)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/server/stats")
    def server_stats():
        return {"execute_limiter": execute_limiter.stats()}

    @app.get("/cache/stats")
    def cache_stats():
        if response_cache is None:
            return {"enabled": False}
        return response_cache.stats()

    @app.post("/cache/clear")
    def cache_clear():
        if response_cache is None:
            return {"cleared": False, "enabled": False}
        response_cache.clear()
        return {"cleared": True, "enabled": True}

    @app.post("/execute")
    def execute(request: ToolRequest):
        tool_name = (request.tool_name or request.tool or "").strip()
        if not tool_name:
            return {"error": "Missing 'tool_name' in request."}

        arguments = request.arguments if isinstance(request.arguments, dict) else {}
        if not arguments and isinstance(request.params, dict):
            arguments = request.params

        image_path = request.image_path or "."
        cache_key = ""
        if response_cache is not None and response_cache.enabled:
            cache_key = _build_cache_key(tool_name, arguments, str(image_path))
            cached = response_cache.get(cache_key)
            if cached is not None:
                return {"output": cached}

        def _run_execute() -> Any:
            return executor.execute(
                tool_name=tool_name,
                params=arguments,
                image_path=Path(str(image_path)),
            )

        result = execute_limiter.run(_run_execute)
        output = "" if result is None else str(result)
        if response_cache is not None and response_cache.enabled:
            is_error = output.strip().startswith("Error")
            response_cache.set(cache_key, output, is_error=is_error)
        return {"output": output}

    return app


def run_server(
    *,
    host: str,
    port: int,
    device: str,
    preload_tools: str | Sequence[str] | None,
    preload_models_list: str | Sequence[str] | None,
    access_log: bool,
    cache_enabled: bool,
    cache_max_entries: int,
    cache_ttl_secs: float,
    cache_errors: bool,
    max_concurrent_executes: int,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    executor = build_executor(device=device)
    preload_executor_tools(executor, _parse_preload_list(preload_tools))
    preload_models(device, _parse_preload_list(preload_models_list))
    response_cache = ResponseCache(
        enabled=cache_enabled,
        max_entries=cache_max_entries,
        ttl_secs=cache_ttl_secs,
        cache_errors=cache_errors,
    )
    app = create_app(
        executor,
        response_cache=response_cache,
        max_concurrent_executes=max_concurrent_executes,
    )
    uvicorn.run(app, host=host, port=port, access_log=access_log)


def get_default_server_kwargs() -> Dict[str, Any]:
    server_cfg = config_loader.get_server_config()
    return {
        "host": server_cfg.host,
        "port": server_cfg.port,
        "device": server_cfg.device,
        "preload_tools": list(server_cfg.preload_tools),
        "preload_models_list": list(server_cfg.preload_models),
        "access_log": server_cfg.access_log,
        "cache_enabled": server_cfg.cache_enabled,
        "cache_max_entries": server_cfg.cache_max_entries,
        "cache_ttl_secs": server_cfg.cache_ttl_secs,
        "cache_errors": server_cfg.cache_errors,
        "max_concurrent_executes": server_cfg.max_concurrent_executes,
    }
