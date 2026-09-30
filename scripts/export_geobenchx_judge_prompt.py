"""Export GeoBenchX's judge prompt with the tool descriptions filled in.

Runs with GeoBenchX's own interpreter (importing ``geobenchx.evaluation`` imports
geopandas and the LLM client libraries)::

    third_party/venvs/geobenchx/bin/python scripts/export_geobenchx_judge_prompt.py OUT.txt

The written template still contains ``{task_text}``, ``{reference_solutions}`` and
``{candidate_solution}``. ``agentic_evals`` reads the file so it never imports
GeoBenchX. Upstream builds this string at import
(``geobenchx/evaluation.py``, ``RESULT_CHECKING_PROMPT``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
# tools.py reads these at import and does not open files until a tool is called.
os.environ.setdefault("STATDATAPATH", "")
os.environ.setdefault("GEODATAPATH", "")

root = Path(os.environ.get("GEOBENCHX_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "GeoBenchX"))
sys.path.insert(0, str(root))

from geobenchx.evaluation import RESULT_CHECKING_PROMPT  # noqa: E402


def main(out: Path) -> None:
    for field in ("{task_text}", "{reference_solutions}", "{candidate_solution}"):
        if field not in RESULT_CHECKING_PROMPT:
            raise SystemExit(f"judge prompt is missing {field}; upstream evaluation.py changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(RESULT_CHECKING_PROMPT, encoding="utf-8")
    print(f"wrote {out} ({len(RESULT_CHECKING_PROMPT)} chars)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
