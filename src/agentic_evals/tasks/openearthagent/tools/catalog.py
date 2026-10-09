"""Tool schemas the end-to-end agent sees.

Argument names and the one-line behavior follow ``RS_AGENT_PROMPT`` at the
pinned revision. The model calls these as native tools. ``Terminate`` is not
a tool: the final assistant message is the answer (see ``PORTING.md``).
Image, GeoPackage, and GeoTIFF arguments are placeholders (``img_1``,
``gpkg_1``, ``tif_1``); the MCP server replaces them from per-sample state
the way ``BaseToolInferencer`` does.
"""

from __future__ import annotations

from typing import Any


def _schema(properties: dict[str, dict[str, str]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


def _str(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _num(description: str) -> dict[str, str]:
    return {"type": "number", "description": description}


def _bool(description: str) -> dict[str, str]:
    return {"type": "boolean", "description": description}


def _obj(description: str) -> dict[str, str]:
    return {"type": "object", "description": description}


def _arr(description: str) -> dict[str, Any]:
    return {"type": "array", "description": description, "items": {"type": "string"}}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "Calculator",
        "description": "Evaluate a Python math expression (operators, math functions, min, max).",
        "parameters": _schema({"expression": _str("Expression, e.g. max(10, 2*sqrt(9)).")}, ["expression"]),
    },
    {
        "name": "OCR",
        "description": "Extract text and bounding boxes from an image.",
        "parameters": _schema({"image": _str("Image handle, usually img_1.")}, ["image"]),
    },
    {
        "name": "DrawBox",
        "description": "Draw a bounding box on an image, optionally labeled.",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "bbox": _str("Box as '(x1,y1,x2,y2)'."),
                "annotation": _str("Optional label."),
            },
            ["image", "bbox"],
        ),
    },
    {
        "name": "AddText",
        "description": "Overlay text at a point or a preset position (lt, mm, rb).",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "text": _str("Text to draw."),
                "position": _str("Coordinates or lt, mm, rb."),
            },
            ["image", "text", "position"],
        ),
    },
    {
        "name": "GoogleSearch",
        "description": "Search the web.",
        "parameters": _schema({"query": _str("Search query."), "k": _num("Optional result count.")}, ["query"]),
    },
    {
        "name": "Plot",
        "description": "Run Python that defines solution() and returns a matplotlib figure. Do not load images.",
        "parameters": _schema({"command": _str("Python source, usually in a ```python fence.")}, ["command"]),
    },
    {
        "name": "Solver",
        "description": "Run SymPy code. solution() must return the result as a string.",
        "parameters": _schema({"command": _str("Python source, usually in a ```python fence.")}, ["command"]),
    },
    {
        "name": "TextToBbox",
        "description": "Detect objects described by text. Returns boxes and confidences.",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "text": _str("What to detect."),
                "top1": _bool("If true, return only the top box."),
            },
            ["image", "text"],
        ),
    },
    {
        "name": "ImageDescription",
        "description": "Describe an image.",
        "parameters": _schema({"image": _str("Image handle.")}, ["image"]),
    },
    {
        "name": "RegionAttributeDescription",
        "description": "Describe an attribute of an image, optionally inside a box.",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "attribute": _str("Attribute to describe."),
                "bbox": _str("Optional '(x1,y1,x2,y2)'."),
            },
            ["image", "attribute"],
        ),
    },
    {
        "name": "CountGivenObject",
        "description": "Count objects, optionally inside a box.",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "text": _str("Object name."),
                "bbox": _str("Optional '(x1,y1,x2,y2)'."),
            },
            ["image", "text"],
        ),
    },
    {
        "name": "ChangeDetection",
        "description": "Compare a pre-event and a post-event image.",
        "parameters": _schema(
            {
                "text": _str("What change to identify."),
                "pre_image": _str("Pre-event image handle."),
                "post_image": _str("Post-event image handle."),
            },
            ["text", "pre_image", "post_image"],
        ),
    },
    {
        "name": "SegmentObjectPixels",
        "description": "Segment objects and return pixel counts. flag true is per object; false is the total.",
        "parameters": _schema(
            {
                "image": _str("Image handle."),
                "text": _str("Object name."),
                "flag": _bool("True for a per-object list."),
            },
            ["image", "text", "flag"],
        ),
    },
    {
        "name": "ObjectDetection",
        "description": "Detect objects. Returns boxes, labels, and confidences.",
        "parameters": _schema({"image": _str("Image handle.")}, ["image"]),
    },
    {
        "name": "GetAreaBoundary",
        "description": "Create a GeoPackage of an area boundary from a place name or box, with an optional buffer.",
        "parameters": _schema(
            {"area": _str("Place name."), "buffer_m": _num("Optional buffer in meters.")},
            ["area"],
        ),
    },
    {
        "name": "AddPoisLayer",
        "description": "Add points of interest inside the area boundary.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle, usually gpkg_1."),
                "query": _obj("OSM key/value filter, e.g. {\"amenity\": \"hospital\"}."),
                "layer_name": _str("Layer to write."),
            },
            ["gpkg", "query", "layer_name"],
        ),
    },
    {
        "name": "ComputeDistance",
        "description": "Measure distances between two layers and save a line layer.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle."),
                "src_layer": _str("Source layer."),
                "tar_layer": _str("Target layer."),
                "top": _num("Optional maximum number of pairs."),
            },
            ["gpkg", "src_layer", "tar_layer"],
        ),
    },
    {
        "name": "DisplayOnMap",
        "description": "Render GeoPackage layers on a basemap.",
        "parameters": _schema(
            {"gpkg": _str("GeoPackage handle."), "layers": _arr("Layer names.")},
            ["gpkg", "layers"],
        ),
    },
    {
        "name": "AddIndexLayer",
        "description": "Compute NDVI, NDBI, or NBR for a year and optional month, and report class percentages.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle."),
                "index_type": _str("NDVI, NDBI, or NBR."),
                "layer_name": _str("Layer to write."),
                "year": _num("Year."),
                "month": _num("Optional month."),
            },
            ["gpkg", "index_type", "layer_name", "year"],
        ),
    },
    {
        "name": "ComputeIndexChange",
        "description": "Compute the difference of two index layers and report change percentages.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle."),
                "index_type": _str("NDVI, NDBI, or NBR."),
                "layer1_name": _str("Earlier layer."),
                "layer2_name": _str("Later layer."),
                "diff_layer_name": _str("Layer to write."),
            },
            ["gpkg", "index_type", "layer1_name", "layer2_name", "diff_layer_name"],
        ),
    },
    {
        "name": "ShowIndexLayer",
        "description": "Write a colorized PNG preview of an index layer.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle."),
                "index_type": _str("NDVI, NDBI, or NBR."),
                "layer_name": _str("Layer to preview."),
            },
            ["gpkg", "index_type", "layer_name"],
        ),
    },
    {
        "name": "GetBboxFromGeotiff",
        "description": "Read an area bounding box (west, south, east, north) from a GeoTIFF.",
        "parameters": _schema({"geotiff": _str("GeoTIFF handle, usually tif_1.")}, ["geotiff"]),
    },
    {
        "name": "DisplayOnGeotiff",
        "description": "Render GeoPackage layers over a GeoTIFF.",
        "parameters": _schema(
            {
                "gpkg": _str("GeoPackage handle."),
                "layers": _arr("Layer names."),
                "geotiff": _str("GeoTIFF handle."),
            },
            ["gpkg", "layers", "geotiff"],
        ),
    },
]


def schemas_for() -> list[dict[str, Any]]:
    return list(TOOL_SCHEMAS)
