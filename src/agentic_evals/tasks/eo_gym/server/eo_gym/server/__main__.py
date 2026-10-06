from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from eo_gym.runtime.core import config_loader
from eo_gym.server.app import _parse_bool, get_default_server_kwargs, run_server


def _env_or_default(env_name: str, fallback: Any) -> Any:
    raw = os.getenv(env_name)
    return raw if raw is not None else fallback


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the EO Gym tool execution API server.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_loader.CONFIG_PATH or config_loader.DEFAULT_CONFIG_PATH),
        help="Path to config/paths.toml.",
    )
    parser.add_argument("--host", type=str, default=_env_or_default("EO_TOOL_SERVER_HOST", defaults["host"]), help="Server host.")
    parser.add_argument("--port", type=int, default=int(_env_or_default("EO_TOOL_SERVER_PORT", defaults["port"])), help="Server port.")
    parser.add_argument("--device", type=str, default=_env_or_default("EO_TOOL_SERVER_DEVICE", defaults["device"]), help="Tool device, e.g. cpu or cuda.")
    parser.add_argument(
        "--preload-tools",
        type=str,
        default=_env_or_default("EO_TOOL_SERVER_PRELOAD_TOOLS", ",".join(defaults["preload_tools"])),
        help="Comma-separated tool names to preload at startup.",
    )
    parser.add_argument(
        "--preload-models",
        type=str,
        default=_env_or_default("EO_TOOL_SERVER_PRELOAD_MODELS", ",".join(defaults["preload_models_list"])),
        help="Comma-separated model preload list: none|all|sam3|dino|ai_optical_analysis|optical_sar.",
    )
    parser.add_argument(
        "--access-log",
        type=_parse_bool,
        default=_parse_bool(_env_or_default("EO_TOOL_SERVER_ACCESS_LOG", defaults["access_log"])),
        help="Enable or disable uvicorn access logs.",
    )
    parser.add_argument(
        "--cache-enabled",
        type=_parse_bool,
        default=_parse_bool(_env_or_default("EO_TOOL_SERVER_CACHE_ENABLED", defaults["cache_enabled"])),
        help="Enable response cache for /execute.",
    )
    parser.add_argument(
        "--cache-max-entries",
        type=int,
        default=int(_env_or_default("EO_TOOL_SERVER_CACHE_MAX_ENTRIES", defaults["cache_max_entries"])),
        help="Max in-memory cached responses.",
    )
    parser.add_argument(
        "--cache-ttl-secs",
        type=float,
        default=float(_env_or_default("EO_TOOL_SERVER_CACHE_TTL_SECS", defaults["cache_ttl_secs"])),
        help="TTL for cached responses in seconds.",
    )
    parser.add_argument(
        "--cache-errors",
        type=_parse_bool,
        default=_parse_bool(_env_or_default("EO_TOOL_SERVER_CACHE_ERRORS", defaults["cache_errors"])),
        help="Cache error responses.",
    )
    parser.add_argument(
        "--max-concurrent-executes",
        type=int,
        default=int(_env_or_default("EO_TOOL_SERVER_MAX_CONCURRENT_EXECUTES", defaults["max_concurrent_executes"])),
        help="Maximum concurrent executor calls in /execute.",
    )
    return parser


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", type=str, default=os.getenv("EO_GYM_CONFIG_FILE"))
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    config_path = bootstrap_args.config or str(config_loader.DEFAULT_CONFIG_PATH)
    config_loader.bootstrap_config(Path(config_path).expanduser())
    defaults = get_default_server_kwargs()
    parser = _build_parser(defaults)
    args = parser.parse_args()
    config_loader.bootstrap_config(Path(args.config).expanduser())

    run_server(
        host=args.host,
        port=args.port,
        device=args.device,
        preload_tools=args.preload_tools,
        preload_models_list=args.preload_models,
        access_log=args.access_log,
        cache_enabled=args.cache_enabled,
        cache_max_entries=args.cache_max_entries,
        cache_ttl_secs=args.cache_ttl_secs,
        cache_errors=args.cache_errors,
        max_concurrent_executes=args.max_concurrent_executes,
    )


if __name__ == "__main__":
    main()
