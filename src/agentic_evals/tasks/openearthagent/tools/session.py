"""Per-sample tool state for one OpenEarthAgent trajectory.

``BaseToolInferencer.batch_get_tool_response`` overwrites ``image``,
``pre_image``, ``post_image``, ``gpkg``, and ``geotiff`` from the running
item. The handles in the prompt (``img_1``, ``gpkg_1``, ``tif_1``) are not
paths. This session is that item: one MCP process serves one sample, so the
state is not shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_evals.tasks.openearthagent.metrics import GPKG_REQUIRED, IMAGE_REQUIRED

MAX_TOOL_TEXT_CHARS = 3000
_INJECTED = ("image", "pre_image", "post_image", "gpkg")


@dataclass
class ToolSession:
    images: list[str]
    sample_num: str
    img: str | None = None
    gpkg: str | None = None
    cached_gpkg: str | None = None
    cached_layers: dict[str, str] = field(default_factory=dict)


def prepare_call(session: ToolSession, name: str, arguments: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(params, error)``. ``error`` is set when the call cannot be sent.

    The error dict matches the inferencer's ``{"text", "error_code": 1}``
    messages for a missing image or GeoPackage. Unknown tool names get the
    same "no tool named" text.
    """
    from agentic_evals.tasks.openearthagent.metrics import TOOLS

    if name not in TOOLS or name == "Terminate":
        return None, {"text": f"There is no tool named {name}.", "error_code": 1}
    params = dict(arguments or {})
    for key in _INJECTED:
        params.pop(key, None)
    params.pop("geotiff", None)

    pre_image = post_image = None
    if name == "ChangeDetection":
        if len(session.images) < 2:
            return None, {"text": "Valid image is required for tool", "error_code": 1}
        pre_image, post_image = session.images[0], session.images[1]

    img = session.img or (session.images[0] if session.images else None)
    if name in IMAGE_REQUIRED and name != "ChangeDetection" and not img:
        return None, {"text": "Valid image is required for tool", "error_code": 1}
    if name in GPKG_REQUIRED and not session.gpkg:
        return None, {"text": f"GeoPackage is required for tool: {name}", "error_code": 1}

    if name == "ChangeDetection":
        params["pre_image"] = pre_image
        params["post_image"] = post_image
    if name in {"DisplayOnGeotiff", "GetBboxFromGeotiff"}:
        params["geotiff"] = img
    if name in IMAGE_REQUIRED and name != "ChangeDetection":
        params["image"] = img
    elif name == "ChangeDetection" and img:
        params["image"] = img
    if name in GPKG_REQUIRED:
        params["gpkg"] = session.gpkg
    return params, None


def observe(session: ToolSession, name: str, response: dict[str, Any]) -> str:
    """Text the model sees. Paths stay on the session, as the inferencer pops them.

    Longer than ``MAX_TOOL_TEXT_CHARS`` is cut with the released ``" ... "`` suffix.
    """
    gpkg = response.get("gpkg")
    image = response.get("image")
    if isinstance(gpkg, str) and gpkg:
        session.gpkg = gpkg
    if isinstance(image, str) and image:
        session.img = image
    text = response.get("text")
    if not isinstance(text, str):
        text = "" if response.get("error_code") == 0 else f"Tool {name} failed"
    if len(text) > MAX_TOOL_TEXT_CHARS:
        text = text[:MAX_TOOL_TEXT_CHARS] + " ... "
    return text
