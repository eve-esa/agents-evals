"""
Runtime configuration loader for EO Gym.

The module exposes package-level path globals used by the tool runtime. Values
are initialized from ``config/paths.toml`` and can be overridden with
``EO_GYM_<KEY>`` environment variables.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# parents: core, runtime, eo_gym, server. Upstream counted one more level
# because the package lived under ``src/``.
REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "paths.toml"


def _env_flag(name: str, default: bool) -> bool:
    for env_name in (f"EO_GYM_{name}", name):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_int(value: Any) -> int:
    return int(value)


def _parse_float(value: Any) -> float:
    return float(value)


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        loaded = tomllib.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _env_value(key: str) -> Optional[str]:
    return os.getenv(f"EO_GYM_{key.upper()}")


def _resolve_value(
    config_data: Dict[str, Any],
    section: str,
    key: str,
    default: Any,
    parser: Callable[[Any], Any],
    *,
    legacy_env: Optional[str] = None,
) -> Any:
    env_raw = _env_value(key)
    if env_raw is not None:
        return parser(env_raw)
    if legacy_env:
        legacy_raw = os.getenv(legacy_env)
        if legacy_raw is not None:
            return parser(legacy_raw)
    section_data = config_data.get(section, {})
    if isinstance(section_data, dict) and key in section_data:
        return parser(section_data[key])
    return parser(default)


def _resolve_local_data_root(config_data: Dict[str, Any]) -> Path:
    env_raw = _env_value("local_data_root")
    if env_raw is not None:
        return _parse_path(env_raw)
    legacy_raw = os.getenv("LOCAL_DATA_ROOT")
    if legacy_raw is not None:
        return _parse_path(legacy_raw)

    paths_data = config_data.get("paths", {})
    if not isinstance(paths_data, dict) or "local_data_root" not in paths_data:
        return DATA_DIR

    local_data_root = _parse_path(paths_data["local_data_root"])
    if "data_dir" in paths_data and local_data_root == _parse_path(paths_data["data_dir"]):
        return DATA_DIR
    return local_data_root


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    device: str
    preload_tools: list[str]
    preload_models: list[str]
    access_log: bool
    cache_enabled: bool
    cache_max_entries: int
    cache_ttl_secs: float
    cache_errors: bool
    max_concurrent_executes: int


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Optional[Path]
    data_dir: Path
    local_data_root: Path
    image_cache_dir: Path
    vqa_outputs_dir: Path
    vqa_mask_mapping_dir: Path
    cropped_to_base_path: Path
    cropped_to_cropped_gt_path: Path
    image_ground_truth_map_path: Path
    image_size_and_gsd_mapping_path: Path
    eo_gym_dataset_csv: Path
    eo_gym_fair1m_ms_csv: Path
    eo_gym_fmow_csv: Path
    eo_gym_m4_sar_csv: Path
    public_data_base_url: str
    fmow_data_root: Path
    fmow_meta_root: Path
    fmow_index_path: Path
    fair1m_index_path: Path
    sam3_model_id: str
    local_vlm_model_id_or_path: str
    bbox_match_model_id_or_path: str
    local_vlm_local_files_only: bool
    server: ServerConfig
    run_with_masking: bool
    run_on_local: bool
    tool_run_on_local: bool


CONFIG_PATH: Optional[Path] = None
RUNTIME_CONFIG: Optional[RuntimeConfig] = None

# Compatibility globals populated by bootstrap_config().
DATA_DIR: Path = REPO_ROOT / "DATA"
IMAGE_CACHE_DIR: Path = REPO_ROOT / "image_cache"
VQA_OUTPUTS_DIR: Path = REPO_ROOT / "VQA_outputs"
VQA_MASK_MAPPING_DIR: Path = VQA_OUTPUTS_DIR / "mask_mapping"
RUN_WITH_MASKING: bool = True
# This copy stores imagery outside the server tree. Local relative paths
# (EO_GYM_DATA/...) are the ones the harness sends. Set EO_GYM_RUN_ON_LOCAL=0
# to restore the upstream remote-URL default.
RUN_ON_LOCAL: bool = True
TOOL_RUN_ON_LOCAL: bool = True
LOCAL_DATA_ROOT: Path = DATA_DIR
MAPPING_DIR: Path = VQA_MASK_MAPPING_DIR
CROPPED_TO_BASE_PATH: Path = MAPPING_DIR / "masking_cropped_image_to_base_images.jsonl"
CROPPED_TO_CROPPED_GT_PATH: Path = MAPPING_DIR / "masking_cropped_image_to_cropped_gt.jsonl"
IMAGE_GROUND_TRUTH_MAP_PATH: Path = MAPPING_DIR / "masking_image_ground_truth_map.jsonl"
IMAGE_SIZE_AND_GSD_MAPPING_PATH: Path = MAPPING_DIR / "masking_image_size_and_gsd_mapping.json"
EO_GYM_DATASET_CSV: Path = MAPPING_DIR / "masking_eo_gym_dataset.csv"
EO_GYM_FAIR1M_MS_CSV: Path = MAPPING_DIR / "masking_eo_gym_fair1m_ms.csv"
EO_GYM_FMOW_CSV: Path = MAPPING_DIR / "masking_eo_gym_fmow.csv"
EO_GYM_M4_SAR_CSV: Path = MAPPING_DIR / "masking_eo_gym_m4_sar.csv"
PUBLIC_DATA_BASE_URL: str = "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/dea-reasoning/"
FMOW_DATA_ROOT: Path = DATA_DIR / "fmow"
FMOW_META_ROOT: Path = REPO_ROOT / "fmow" / "seq"
FMOW_VQA_PATH: Path = VQA_OUTPUTS_DIR / "fmow_balanced_vqa_mcq.jsonl"
FMOW_INDEX_PATH: Path = DATA_DIR / "fmow_image_index.json"
M4_SAR_VQA_PATH: Path = VQA_OUTPUTS_DIR / "m4_sar_balanced_vqa_mcq.jsonl"
FAIR1M_INDEX_PATH: Path = DATA_DIR / "fair1m_multispectral_index.json"
SAM3_MODEL_ID: str = "facebook/sam3"
LOCAL_VLM_MODEL_ID_OR_PATH: str = "Qwen/Qwen2.5-VL-3B-Instruct"
BBOX_MATCH_MODEL_ID_OR_PATH: str = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_VLM_LOCAL_FILES_ONLY: bool = True
SERVER_CONFIG: ServerConfig


def bootstrap_config(config_path: Optional[str | Path] = None) -> RuntimeConfig:
    global CONFIG_PATH
    global RUNTIME_CONFIG
    global DATA_DIR
    global IMAGE_CACHE_DIR
    global VQA_OUTPUTS_DIR
    global VQA_MASK_MAPPING_DIR
    global RUN_WITH_MASKING
    global RUN_ON_LOCAL
    global TOOL_RUN_ON_LOCAL
    global LOCAL_DATA_ROOT
    global MAPPING_DIR
    global CROPPED_TO_BASE_PATH
    global CROPPED_TO_CROPPED_GT_PATH
    global IMAGE_GROUND_TRUTH_MAP_PATH
    global IMAGE_SIZE_AND_GSD_MAPPING_PATH
    global EO_GYM_DATASET_CSV
    global EO_GYM_FAIR1M_MS_CSV
    global EO_GYM_FMOW_CSV
    global EO_GYM_M4_SAR_CSV
    global PUBLIC_DATA_BASE_URL
    global FMOW_DATA_ROOT
    global FMOW_META_ROOT
    global FMOW_VQA_PATH
    global FMOW_INDEX_PATH
    global M4_SAR_VQA_PATH
    global FAIR1M_INDEX_PATH
    global SAM3_MODEL_ID
    global LOCAL_VLM_MODEL_ID_OR_PATH
    global BBOX_MATCH_MODEL_ID_OR_PATH
    global LOCAL_VLM_LOCAL_FILES_ONLY
    global SERVER_CONFIG

    selected_path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path(os.getenv("EO_GYM_CONFIG_FILE", DEFAULT_CONFIG_PATH)).expanduser()
    )
    if not selected_path.is_absolute():
        selected_path = (REPO_ROOT / selected_path).resolve()
    config_data = _load_toml(selected_path)
    CONFIG_PATH = selected_path if selected_path.exists() else None

    RUN_WITH_MASKING = _env_flag("RUN_WITH_MASKING", True)
    RUN_ON_LOCAL = _env_flag("RUN_ON_LOCAL", True)
    TOOL_RUN_ON_LOCAL = _env_flag("TOOL_RUN_ON_LOCAL", True)

    DATA_DIR = _resolve_value(config_data, "paths", "data_dir", REPO_ROOT / "DATA", _parse_path, legacy_env="DEA_DATA_DIR")
    LOCAL_DATA_ROOT = _resolve_local_data_root(config_data)
    IMAGE_CACHE_DIR = _resolve_value(config_data, "paths", "image_cache_dir", REPO_ROOT / "image_cache", _parse_path, legacy_env="IMAGE_CACHE_DIR")
    VQA_OUTPUTS_DIR = _resolve_value(config_data, "paths", "vqa_outputs_dir", REPO_ROOT / "VQA_outputs", _parse_path, legacy_env="VQA_OUTPUTS_DIR")
    VQA_MASK_MAPPING_DIR = _resolve_value(
        config_data,
        "paths",
        "vqa_mask_mapping_dir",
        VQA_OUTPUTS_DIR / "mask_mapping",
        _parse_path,
        legacy_env="VQA_MASK_MAPPING_DIR",
    )
    MAPPING_DIR = VQA_MASK_MAPPING_DIR if RUN_WITH_MASKING else VQA_OUTPUTS_DIR / "mapping"

    CROPPED_TO_BASE_PATH = _resolve_value(
        config_data,
        "paths",
        "cropped_to_base_path",
        MAPPING_DIR / ("masking_cropped_image_to_base_images.jsonl" if RUN_WITH_MASKING else "cropped_image_to_base_images.jsonl"),
        _parse_path,
        legacy_env="CROPPED_TO_BASE_PATH",
    )
    CROPPED_TO_CROPPED_GT_PATH = _resolve_value(
        config_data,
        "paths",
        "cropped_to_cropped_gt_path",
        MAPPING_DIR / ("masking_cropped_image_to_cropped_gt.jsonl" if RUN_WITH_MASKING else "cropped_image_to_cropped_gt.jsonl"),
        _parse_path,
        legacy_env="CROPPED_TO_CROPPED_GT_PATH",
    )
    IMAGE_GROUND_TRUTH_MAP_PATH = _resolve_value(
        config_data,
        "paths",
        "image_ground_truth_map_path",
        MAPPING_DIR / ("masking_image_ground_truth_map.jsonl" if RUN_WITH_MASKING else "image_ground_truth_map.jsonl"),
        _parse_path,
        legacy_env="IMAGE_GROUND_TRUTH_MAP_PATH",
    )
    IMAGE_SIZE_AND_GSD_MAPPING_PATH = _resolve_value(
        config_data,
        "paths",
        "image_size_and_gsd_mapping_path",
        MAPPING_DIR / ("masking_image_size_and_gsd_mapping.json" if RUN_WITH_MASKING else "image_size_and_gsd_mapping.json"),
        _parse_path,
        legacy_env="IMAGE_SIZE_AND_GSD_MAPPING_PATH",
    )
    EO_GYM_DATASET_CSV = _resolve_value(
        config_data,
        "paths",
        "eo_gym_dataset_csv",
        MAPPING_DIR / ("masking_eo_gym_dataset.csv" if RUN_WITH_MASKING else "eo_gym_dataset.csv"),
        _parse_path,
        legacy_env="EO_GYM_DATASET_CSV",
    )
    EO_GYM_FAIR1M_MS_CSV = _resolve_value(
        config_data,
        "paths",
        "eo_gym_fair1m_ms_csv",
        MAPPING_DIR / ("masking_eo_gym_fair1m_ms.csv" if RUN_WITH_MASKING else "eo_gym_fair1m_ms.csv"),
        _parse_path,
        legacy_env="EO_GYM_FAIR1M_MS_CSV",
    )
    EO_GYM_FMOW_CSV = _resolve_value(
        config_data,
        "paths",
        "eo_gym_fmow_csv",
        MAPPING_DIR / ("masking_eo_gym_fmow.csv" if RUN_WITH_MASKING else "eo_gym_fmow.csv"),
        _parse_path,
        legacy_env="EO_GYM_FMOW_CSV",
    )
    EO_GYM_M4_SAR_CSV = _resolve_value(
        config_data,
        "paths",
        "eo_gym_m4_sar_csv",
        MAPPING_DIR / ("masking_eo_gym_m4_sar.csv" if RUN_WITH_MASKING else "eo_gym_m4_sar.csv"),
        _parse_path,
        legacy_env="EO_GYM_M4_SAR_CSV",
    )

    PUBLIC_DATA_BASE_URL = str(
        _resolve_value(
            config_data,
            "urls",
            "public_data_base_url",
            "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com/projects/dea-reasoning/",
            str,
            legacy_env="PUBLIC_DATA_BASE_URL",
        )
    )

    FMOW_DATA_ROOT = _resolve_value(config_data, "paths", "fmow_data_root", DATA_DIR / "fmow", _parse_path)
    FMOW_META_ROOT = _resolve_value(config_data, "paths", "fmow_meta_root", REPO_ROOT / "fmow" / "seq", _parse_path)
    FMOW_INDEX_PATH = _resolve_value(config_data, "paths", "fmow_index_path", DATA_DIR / "fmow_image_index.json", _parse_path)
    FAIR1M_INDEX_PATH = _resolve_value(config_data, "paths", "fair1m_index_path", DATA_DIR / "fair1m_multispectral_index.json", _parse_path)
    FMOW_VQA_PATH = VQA_OUTPUTS_DIR / "fmow_balanced_vqa_mcq.jsonl"
    M4_SAR_VQA_PATH = VQA_OUTPUTS_DIR / "m4_sar_balanced_vqa_mcq.jsonl"

    SAM3_MODEL_ID = str(_resolve_value(config_data, "models", "sam3_model_id", "facebook/sam3", str))
    LOCAL_VLM_MODEL_ID_OR_PATH = str(
        _resolve_value(config_data, "models", "local_vlm_model_id_or_path", "Qwen/Qwen2.5-VL-3B-Instruct", str)
    )
    BBOX_MATCH_MODEL_ID_OR_PATH = str(
        _resolve_value(config_data, "models", "bbox_match_model_id_or_path", "Qwen/Qwen2.5-0.5B-Instruct", str)
    )
    LOCAL_VLM_LOCAL_FILES_ONLY = bool(
        _resolve_value(config_data, "models", "local_vlm_local_files_only", True, _parse_bool)
    )

    SERVER_CONFIG = ServerConfig(
        host=str(_resolve_value(config_data, "server", "host", "0.0.0.0", str)),
        port=_resolve_value(config_data, "server", "port", 8081, _parse_int),
        device=str(_resolve_value(config_data, "server", "device", "cuda", str)),
        preload_tools=_resolve_value(config_data, "server", "preload_tools", [], _parse_list),
        preload_models=_resolve_value(config_data, "server", "preload_models", ["all"], _parse_list),
        access_log=bool(_resolve_value(config_data, "server", "access_log", False, _parse_bool)),
        cache_enabled=bool(_resolve_value(config_data, "server", "cache_enabled", True, _parse_bool)),
        cache_max_entries=_resolve_value(config_data, "server", "cache_max_entries", 4096, _parse_int),
        cache_ttl_secs=_resolve_value(config_data, "server", "cache_ttl_secs", 3600.0, _parse_float),
        cache_errors=bool(_resolve_value(config_data, "server", "cache_errors", False, _parse_bool)),
        max_concurrent_executes=_resolve_value(config_data, "server", "max_concurrent_executes", 4, _parse_int),
    )

    RUNTIME_CONFIG = RuntimeConfig(
        config_path=CONFIG_PATH,
        data_dir=DATA_DIR,
        local_data_root=LOCAL_DATA_ROOT,
        image_cache_dir=IMAGE_CACHE_DIR,
        vqa_outputs_dir=VQA_OUTPUTS_DIR,
        vqa_mask_mapping_dir=VQA_MASK_MAPPING_DIR,
        cropped_to_base_path=CROPPED_TO_BASE_PATH,
        cropped_to_cropped_gt_path=CROPPED_TO_CROPPED_GT_PATH,
        image_ground_truth_map_path=IMAGE_GROUND_TRUTH_MAP_PATH,
        image_size_and_gsd_mapping_path=IMAGE_SIZE_AND_GSD_MAPPING_PATH,
        eo_gym_dataset_csv=EO_GYM_DATASET_CSV,
        eo_gym_fair1m_ms_csv=EO_GYM_FAIR1M_MS_CSV,
        eo_gym_fmow_csv=EO_GYM_FMOW_CSV,
        eo_gym_m4_sar_csv=EO_GYM_M4_SAR_CSV,
        public_data_base_url=PUBLIC_DATA_BASE_URL,
        fmow_data_root=FMOW_DATA_ROOT,
        fmow_meta_root=FMOW_META_ROOT,
        fmow_index_path=FMOW_INDEX_PATH,
        fair1m_index_path=FAIR1M_INDEX_PATH,
        sam3_model_id=SAM3_MODEL_ID,
        local_vlm_model_id_or_path=LOCAL_VLM_MODEL_ID_OR_PATH,
        bbox_match_model_id_or_path=BBOX_MATCH_MODEL_ID_OR_PATH,
        local_vlm_local_files_only=LOCAL_VLM_LOCAL_FILES_ONLY,
        server=SERVER_CONFIG,
        run_with_masking=RUN_WITH_MASKING,
        run_on_local=RUN_ON_LOCAL,
        tool_run_on_local=TOOL_RUN_ON_LOCAL,
    )
    return RUNTIME_CONFIG


def get_server_config() -> ServerConfig:
    if RUNTIME_CONFIG is None:
        bootstrap_config()
    assert SERVER_CONFIG is not None
    return SERVER_CONFIG


def get_runtime_config() -> RuntimeConfig:
    if RUNTIME_CONFIG is None:
        return bootstrap_config()
    return RUNTIME_CONFIG


def as_dict() -> Dict[str, Any]:
    cfg = get_runtime_config()
    return {
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "paths": {
            "data_dir": str(cfg.data_dir),
            "local_data_root": str(cfg.local_data_root),
            "image_cache_dir": str(cfg.image_cache_dir),
            "vqa_outputs_dir": str(cfg.vqa_outputs_dir),
            "vqa_mask_mapping_dir": str(cfg.vqa_mask_mapping_dir),
            "cropped_to_base_path": str(cfg.cropped_to_base_path),
            "cropped_to_cropped_gt_path": str(cfg.cropped_to_cropped_gt_path),
            "image_ground_truth_map_path": str(cfg.image_ground_truth_map_path),
            "image_size_and_gsd_mapping_path": str(cfg.image_size_and_gsd_mapping_path),
            "eo_gym_dataset_csv": str(cfg.eo_gym_dataset_csv),
            "eo_gym_fair1m_ms_csv": str(cfg.eo_gym_fair1m_ms_csv),
            "eo_gym_fmow_csv": str(cfg.eo_gym_fmow_csv),
            "eo_gym_m4_sar_csv": str(cfg.eo_gym_m4_sar_csv),
            "fmow_data_root": str(cfg.fmow_data_root),
            "fmow_meta_root": str(cfg.fmow_meta_root),
            "fmow_index_path": str(cfg.fmow_index_path),
            "fair1m_index_path": str(cfg.fair1m_index_path),
        },
        "models": {
            "sam3_model_id": cfg.sam3_model_id,
            "local_vlm_model_id_or_path": cfg.local_vlm_model_id_or_path,
            "bbox_match_model_id_or_path": cfg.bbox_match_model_id_or_path,
            "local_vlm_local_files_only": cfg.local_vlm_local_files_only,
        },
        "urls": {
            "public_data_base_url": cfg.public_data_base_url,
        },
        "server": {
            "host": cfg.server.host,
            "port": cfg.server.port,
            "device": cfg.server.device,
            "preload_tools": cfg.server.preload_tools,
            "preload_models": cfg.server.preload_models,
            "access_log": cfg.server.access_log,
            "cache_enabled": cfg.server.cache_enabled,
            "cache_max_entries": cfg.server.cache_max_entries,
            "cache_ttl_secs": cfg.server.cache_ttl_secs,
            "cache_errors": cfg.server.cache_errors,
            "max_concurrent_executes": cfg.server.max_concurrent_executes,
        },
        "flags": {
            "run_with_masking": cfg.run_with_masking,
            "run_on_local": cfg.run_on_local,
            "tool_run_on_local": cfg.tool_run_on_local,
        },
    }


def _strip_query(value: str) -> str:
    return str(value).split("?", 1)[0]


def _imagery_candidates(rel: Path) -> list[Path]:
    """Paths for an ``EO_GYM_DATA/...`` suffix.

    ``data_dir`` in this copy is the imagery directory itself. Prefer a file
    that is already there, otherwise the ``data_dir`` location, so a missing
    image is reported under ``data/eo_gym`` and not under ``server/``.
    """
    candidates: list[Path] = []
    if DATA_DIR.name == "EO_GYM_DATA":
        tail = Path(*rel.parts[1:]) if rel.parts and rel.parts[0] == "EO_GYM_DATA" else rel
        candidates.append((DATA_DIR / tail).resolve())
    candidates.append((REPO_ROOT / rel).resolve())
    return candidates


def _prefer_imagery(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _resolve_eo_gym_data_path(path: Path) -> Optional[Path]:
    parts = path.parts
    if "EO_GYM_DATA" not in parts:
        return None
    idx = parts.index("EO_GYM_DATA")
    return _prefer_imagery(_imagery_candidates(Path(*parts[idx:])))


def to_local_path(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    if raw.startswith(("http://", "https://")):
        raw = _strip_query(raw)
        base = (PUBLIC_DATA_BASE_URL or "").rstrip("/")
        if base and raw.startswith(base):
            remainder = raw[len(base) :].lstrip("/")
            if remainder.startswith("DATA/"):
                local = LOCAL_DATA_ROOT / remainder[len("DATA/") :]
            else:
                local = REPO_ROOT / remainder
            resolved = local.resolve()
            override = _resolve_eo_gym_data_path(resolved)
            return str(override or resolved)
        if "EO_GYM_DATA" in raw:
            tail = raw.split("EO_GYM_DATA", 1)[1].lstrip("/")
            chosen = _prefer_imagery(_imagery_candidates(Path("EO_GYM_DATA") / tail))
            if chosen is not None:
                return str(chosen)
        return raw
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    override = _resolve_eo_gym_data_path(path)
    return str(override or path)


def to_remote_url(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    if raw.startswith(("http://", "https://")):
        return _strip_query(raw)
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    rel_path = None
    try:
        rel_path = path.relative_to(LOCAL_DATA_ROOT)
        rel_path = Path("DATA") / rel_path
    except ValueError:
        try:
            rel_path = path.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = None
    base = (PUBLIC_DATA_BASE_URL or "").rstrip("/")
    if rel_path is None and base:
        normalized = path.as_posix()
        for anchor in ("/EO_GYM_DATA/", "/DATA/"):
            idx = normalized.find(anchor)
            if idx != -1:
                rel_path = Path(normalized[idx + 1 :])
                break
    if rel_path is None or not base:
        return raw
    return f"{base}/{rel_path.as_posix()}"


@dataclass
class InferenceConfig:
    input_root: Path
    input_glob: str
    output_dest: Path
    assistant_model: str
    review_model: str
    simulator_model: str
    max_iterations: int
    validation_max_attempts: int
    target_trajs: int
    direct_tool_device: str
    tool_max_retries: int
    action_max_retries: int
    num_workers: int
    log_path: Path
    committee_score_threshold: float
    committee_max_attempts: int
    oracle_mode: bool
    run_with_masking: bool = True
    run_on_local: bool = False
    cropped_to_base_path: Path = CROPPED_TO_BASE_PATH
    cropped_to_cropped_gt_path: Path = CROPPED_TO_CROPPED_GT_PATH
    image_ground_truth_map_path: Path = IMAGE_GROUND_TRUTH_MAP_PATH
    image_size_and_gsd_mapping_path: Path = IMAGE_SIZE_AND_GSD_MAPPING_PATH
    repo_root: Path = REPO_ROOT
    data_dir: Path = DATA_DIR
    image_cache_dir: Path = IMAGE_CACHE_DIR


@dataclass
class ToolsConfig:
    input_root: Path
    input_glob: str
    output_dest: Path
    model_name: str
    max_iterations: int
    image_cache_dir: Path
    device: str
    tool_max_retries: int
    action_max_retries: int
    log_path: Path
    repo_root: Path = REPO_ROOT
    data_dir: Path = DATA_DIR
    default_image_cache_dir: Path = IMAGE_CACHE_DIR
    run_with_masking: bool = RUN_WITH_MASKING
    run_on_local: bool = RUN_ON_LOCAL
    cropped_to_base_path: Path = CROPPED_TO_BASE_PATH
    cropped_to_cropped_gt_path: Path = CROPPED_TO_CROPPED_GT_PATH
    image_ground_truth_map_path: Path = IMAGE_GROUND_TRUTH_MAP_PATH
    image_size_and_gsd_mapping_path: Path = IMAGE_SIZE_AND_GSD_MAPPING_PATH


def load_inference_config() -> InferenceConfig:
    _env_input = os.getenv("SIM_INPUT_JSONL") or os.getenv("SIM_INPUT_PATH")
    input_root = Path(_env_input) if _env_input else VQA_OUTPUTS_DIR
    return InferenceConfig(
        input_root=input_root,
        input_glob=os.getenv("SIM_INPUT_GLOB", "*_vqa_mcq.jsonl"),
        output_dest=Path(os.getenv("SIM_OUTPUT_JSONL", "Agent_SIM_outputs")),
        assistant_model=os.getenv("OPENAI_ASSISTANT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1")),
        review_model=os.getenv("OPENAI_REVIEW_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.1")),
        simulator_model=os.getenv("OPENAI_SIMULATOR_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        max_iterations=int(os.getenv("SIM_MAX_ITERATIONS", "15")),
        validation_max_attempts=int(os.getenv("SIM_VALIDATION_MAX_ATTEMPTS", "2")),
        target_trajs=int(os.getenv("SIM_TARGET_TRAJS_PER_QA", "1")),
        direct_tool_device=os.getenv("SIM_DIRECT_TOOL_DEVICE", "cuda"),
        tool_max_retries=int(os.getenv("SIM_TOOL_MAX_RETRIES", "3")),
        action_max_retries=int(os.getenv("SIM_ACTION_MAX_RETRIES", "3")),
        num_workers=int(os.getenv("SIM_NUM_WORKERS", "8")),
        log_path=Path(os.getenv("SIM_LOG_PATH", "Agent_SIM_outputs/agent_simulation.log")),
        committee_score_threshold=float(os.getenv("SIM_COMMITTEE_SCORE_THRESHOLD", "0.9")),
        committee_max_attempts=int(os.getenv("SIM_COMMITTEE_MAX_ATTEMPTS", "3")),
        oracle_mode=_env_flag("ORACLE_MODE", True),
        run_with_masking=RUN_WITH_MASKING,
        run_on_local=RUN_ON_LOCAL,
        cropped_to_base_path=CROPPED_TO_BASE_PATH,
        cropped_to_cropped_gt_path=CROPPED_TO_CROPPED_GT_PATH,
        image_ground_truth_map_path=IMAGE_GROUND_TRUTH_MAP_PATH,
        image_size_and_gsd_mapping_path=IMAGE_SIZE_AND_GSD_MAPPING_PATH,
        data_dir=DATA_DIR,
        image_cache_dir=IMAGE_CACHE_DIR,
    )


def load_tools_config() -> ToolsConfig:
    _env_input = os.getenv("TOOLS_INPUT_JSONL") or os.getenv("TOOLS_INPUT_PATH")
    input_root = Path(_env_input) if _env_input else VQA_OUTPUTS_DIR
    return ToolsConfig(
        input_root=input_root,
        input_glob=os.getenv("TOOLS_INPUT_GLOB", "*_vqa_mcq.jsonl"),
        output_dest=Path(os.getenv("TOOLS_OUTPUT_PATH", "Agent_Tools_outputs")),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        max_iterations=int(os.getenv("TOOLS_MAX_ITERATIONS", "15")),
        image_cache_dir=Path(os.getenv("TOOLS_IMAGE_CACHE_DIR", str(IMAGE_CACHE_DIR))),
        device=os.getenv("TOOLS_DEVICE", "CUDA"),
        tool_max_retries=int(os.getenv("TOOLS_MAX_RETRIES", "2")),
        action_max_retries=int(os.getenv("TOOLS_ACTION_MAX_RETRIES", "2")),
        log_path=Path(os.getenv("TOOLS_LOG_PATH", "Agent_Tools_outputs/agent_tools.log")),
        data_dir=DATA_DIR,
        default_image_cache_dir=IMAGE_CACHE_DIR,
        run_with_masking=RUN_WITH_MASKING,
        run_on_local=RUN_ON_LOCAL,
        cropped_to_base_path=CROPPED_TO_BASE_PATH,
        cropped_to_cropped_gt_path=CROPPED_TO_CROPPED_GT_PATH,
        image_ground_truth_map_path=IMAGE_GROUND_TRUTH_MAP_PATH,
        image_size_and_gsd_mapping_path=IMAGE_SIZE_AND_GSD_MAPPING_PATH,
    )


bootstrap_config()
