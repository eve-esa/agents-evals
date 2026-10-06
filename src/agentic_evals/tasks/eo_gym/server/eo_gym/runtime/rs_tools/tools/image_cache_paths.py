"""
Helpers for returning paths to locally cached image artifacts.
"""

from __future__ import annotations

from pathlib import Path


def cache_local_path(local_path: Path) -> str:
    """
    Return the absolute local path for a cached artifact.

    EO Gym keeps generated images and crop files on the local filesystem; it
    does not upload cache artifacts to external storage.
    """
    return str(Path(local_path).resolve())
