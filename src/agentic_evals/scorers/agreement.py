"""Agreement between binary verdicts stored in an Inspect log.

For the hands-on session's point that a single judged number hides
disagreement: given a log with several scorers (e.g. a benchmark's own accuracy
and one or more ``trajectory_judge`` scorers), report for every pair the raw
agreement and Cohen's kappa over the samples both scored.

Reference: J. Cohen, "A Coefficient of Agreement for Nominal Scales",
Educational and Psychological Measurement 20(1), 1960, doi:10.1177/001316446002000104.
Kappa = (p_o - p_e) / (1 - p_e), with p_e from each rater's marginal rates.

Usage::

    uv run python -m agentic_evals.scorers.agreement logs/<run>.eval \\
        --verdict earth_bench_scorer:accuracy --verdict trajectory_judge:trajectory_accuracy \\
        --verdict trajectory_judge1:trajectory_accuracy
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Sequence

from inspect_ai.log import read_eval_log


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty verdict lists")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def verdicts(log_path: str, specs: Sequence[str]) -> dict[str, dict[str, bool]]:
    """``{"scorer:key": {sample_key: verdict}}`` for each ``scorer:key`` spec."""
    log = read_eval_log(log_path)
    out: dict[str, dict[str, bool]] = {spec: {} for spec in specs}
    for sample in log.samples or []:
        for spec in specs:
            scorer_name, key = spec.split(":", 1)
            score = (sample.scores or {}).get(scorer_name)
            if score is None or not isinstance(score.value, dict):
                continue
            value = score.value.get(key)
            if isinstance(value, (int, float)) and not math.isnan(value):
                out[spec][f"{sample.id}#{sample.epoch}"] = bool(value)
    return out


def pairwise(table: dict[str, dict[str, bool]]) -> list[dict[str, object]]:
    rows = []
    for x, y in itertools.combinations(table, 2):
        common = sorted(set(table[x]) & set(table[y]))
        if not common:
            continue
        a = [table[x][k] for k in common]
        b = [table[y][k] for k in common]
        rows.append(
            {
                "a": x,
                "b": y,
                "n": len(common),
                "agreement": sum(p == q for p, q in zip(a, b)) / len(common),
                "kappa": cohen_kappa(a, b),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--verdict", action="append", required=True, help="scorer_name:value_key")
    args = parser.parse_args()
    for row in pairwise(verdicts(args.log, args.verdict)):
        print(f"{row['a']} vs {row['b']}: n={row['n']} agreement={row['agreement']:.3f} kappa={row['kappa']:.3f}")


if __name__ == "__main__":
    main()
