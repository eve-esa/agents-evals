"""Export EO-Gym's tool schemas and per-dataset tool subsets to JSON.

Runs with EO-Gym's own interpreter (its registry imports torch and friends)::

    third_party/venvs/eo-gym/bin/python scripts/export_eo_gym_tools.py OUT.json

Schemas are built exactly as EO-Gym's evaluation driver builds them
(``eo_gym_evaluation.tool_schema.build_tool_schemas`` over
``AVAILABLE_TOOL_CLASSES``); the "skill" subsets are
``get_recommended_tools_for_dataset(dataset, allow_fallback=False)``, falling back
to all tools as ``driver.resolve_tools`` does. ``agentic_evals`` reads this file
so it never has to import EO-Gym.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RUN_ON_LOCAL", "1")
os.environ.setdefault("TOOL_RUN_ON_LOCAL", "1")

root = Path(os.environ.get("EO_GYM_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "EO-Gym"))
sys.path.insert(0, str(root / "eo-gym-evaluation" / "src"))

from eo_gym.runtime.registry.Tools import AVAILABLE_TOOL_CLASSES, get_recommended_tools_for_dataset  # noqa: E402
from eo_gym_evaluation.tool_schema import build_tool_schemas  # noqa: E402

DATASETS = ["fair1m_ms", "fmow", "m4-sar", "fair1_vhr", "sardet_100k", "dior", "dota", "xview", "xbd"]


def main(out: Path) -> None:
    all_tools = sorted(AVAILABLE_TOOL_CLASSES.keys())
    skill = {}
    for dataset in DATASETS:
        skill[dataset] = get_recommended_tools_for_dataset(dataset, allow_fallback=False) or all_tools
    payload = {
        "all_tools": all_tools,
        "skill_tools": skill,
        "schemas": build_tool_schemas(all_tools, AVAILABLE_TOOL_CLASSES),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"{len(all_tools)} tools, {len(skill)} dataset subsets -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
