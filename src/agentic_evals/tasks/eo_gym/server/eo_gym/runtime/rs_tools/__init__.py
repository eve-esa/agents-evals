"""
Lightweight rs_tools package scaffold.

Exports will be lazily resolved to avoid circular imports with Tools.py.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "AVAILABLE_TOOL_CLASSES",
    "DATASET_TOOL_ALIASES",
    "DATASET_TOOL_RECOMMENDATIONS",
    "get_recommended_tools_for_dataset",
    "load_custom_tool",
]


def __getattr__(name: str) -> Any:
    """
    Lazily delegate registry exports to Tools to avoid circular imports during package init.
    """
    if name in __all__:
        tools = import_module("eo_gym.runtime.registry.Tools")
        return getattr(tools, name)
    raise AttributeError(name)
