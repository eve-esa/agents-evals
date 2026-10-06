"""
Common utilities for rs_tools.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import httpx

from eo_gym.runtime.core.config_loader import (
    EO_GYM_FAIR1M_MS_CSV,
    PUBLIC_DATA_BASE_URL,
    REPO_ROOT,
    RUN_ON_LOCAL,
    to_local_path,
    to_remote_url,
)


def _normalize_http_url(url_str: str) -> str:
    url_str = url_str.strip()
    if url_str.startswith("https:/") and not url_str.startswith("https://"):
        return url_str.replace("https:/", "https://", 1)
    if url_str.startswith("http:/") and not url_str.startswith("http://"):
        return url_str.replace("http:/", "http://", 1)
    return url_str


def download_file(url: str, cache_dir: Path | str, timeout: float = 60.0) -> str:
    """
    Download a file from a URL to a local cache directory.
    Returns the local path as a string.
    """
    url_str = _normalize_http_url(str(url))
    if not url_str.startswith(("http://", "https://")):
        return url_str

    if RUN_ON_LOCAL:
        return to_local_path(url_str)

    base = (PUBLIC_DATA_BASE_URL or "").rstrip("/")
    if base and url_str.startswith(base):
        remainder = url_str[len(base) :].lstrip("/")
        local_candidate = (REPO_ROOT / remainder).resolve()
        if local_candidate.exists():
            return str(local_candidate)
        
    filename = url_str.split("/")[-1].split("?")[0]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_path = cache_path / filename
    
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path.resolve())
        
    print(f"[INFO] Downloading {url_str} to {local_path}...")
    try:
        with httpx.stream("GET", url_str, follow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with local_path.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        return str(local_path.resolve())
    except Exception as exc:
        print(f"[WARN] Download failed for {url_str}: {exc}")
        return url_str


def _local_candidate(path_or_url: str) -> Optional[Path]:
    raw = str(path_or_url or "").strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    mapped = to_local_path(raw)
    if mapped and not str(mapped).startswith(("http://", "https://")):
        candidate = Path(mapped).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate
    if raw.startswith(("http://", "https://")):
        base = (PUBLIC_DATA_BASE_URL or "").rstrip("/")
        if base and raw.startswith(base):
            remainder = raw[len(base) :].lstrip("/")
            repo_candidate = (REPO_ROOT / remainder).resolve()
            if repo_candidate.exists():
                return repo_candidate
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.exists():
        return candidate
    return None


def check_file_reference(
    path_or_url: str,
    *,
    label: str = "image",
    timeout: float = 10.0,
    strict: bool = False,
) -> Optional[str]:
    raw = _normalize_http_url(str(path_or_url or ""))
    if not raw.strip():
        return f"Missing {label} reference. Please check the provided parameters."
    if raw.endswith("/"):
        return f"Expected a {label} file but got a folder. Please check the provided {label} reference."

    local = _local_candidate(raw)
    if local is not None:
        if local.is_dir():
            return f"Expected a {label} file but got a folder. Please check the provided {label} reference."
        return None

    if raw.startswith(("http://", "https://")):
        try:
            with httpx.stream("HEAD", raw, follow_redirects=True, timeout=timeout) as resp:
                status = resp.status_code
            if status == 405:
                with httpx.stream("GET", raw, follow_redirects=True, timeout=timeout) as resp:
                    status = resp.status_code
            if status >= 400:
                return f"{label.capitalize()} URL not found. Please check the provided {label} reference."
            return None
        except Exception:
            if strict:
                return f"Unable to verify the provided {label} URL. Please check the {label} reference."
            return None

    return f"{label.capitalize()} file not found. Please check the provided {label} path."


def _normalize_folder_value(value: str) -> str:
    raw = str(value or "").strip()
    if raw.endswith("/"):
        raw = raw.rstrip("/")
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    return raw


def _candidate_folder_keys(path_or_url: str) -> list[str]:
    raw = _normalize_folder_value(path_or_url)
    if not raw:
        return []
    candidates = {raw}
    candidates.add(_normalize_folder_value(to_remote_url(raw)))
    candidates.add(_normalize_folder_value(to_local_path(raw)))
    return [c for c in candidates if c]


@lru_cache(maxsize=1)
def _load_ms_folder_index() -> Optional[Set[str]]:
    if not EO_GYM_FAIR1M_MS_CSV.exists():
        return None
    entries: Set[str] = set()
    try:
        with EO_GYM_FAIR1M_MS_CSV.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                folder_path = _normalize_folder_value(row.get("folder_path") or "")
                if folder_path:
                    entries.add(folder_path)
    except Exception:
        return None
    return entries


def check_multispectral_folder_reference(
    ms_folder: str,
    *,
    label: str = "multispectral folder",
) -> Optional[str]:
    raw = _normalize_folder_value(ms_folder)
    if not raw:
        return f"Missing {label} reference. Please check the provided parameters."
    index = _load_ms_folder_index()
    if not index:
        return "Multispectral index file not found. Please check your setup."
    candidates = _candidate_folder_keys(raw)
    for candidate in candidates:
        if candidate in index:
            return None
    return f"{label.capitalize()} not found. Please check the provided {label} reference."
