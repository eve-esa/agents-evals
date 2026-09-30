"""Tasks, agents and scorers exposed to Inspect (``[project.entry-points.inspect_ai]``).

Anything imported here is addressable by name, e.g.
``inspect eval agentic_evals/earth_bench --solver agentic_evals/langgraph_react``.
"""

# ruff: noqa: F401

from agentic_evals.agents.langgraph_react import langgraph_react
from agentic_evals.max_tokens import UncappedMaxTokens as _uncapped_max_tokens
from agentic_evals.openai_compat import install as _install_openai_compat
from agentic_evals.scorers.judges import trajectory_judge
from agentic_evals.tasks.earth_bench.scorers import earth_bench_scorer
from agentic_evals.tasks.earth_bench.task import earth_bench
from agentic_evals.tasks.eo_gym.scorers import eo_gym_early_stop, eo_gym_pooled, eo_gym_scorer
from agentic_evals.tasks.eo_gym.task import eo_gym
from agentic_evals.tasks.geobenchx.scorers import geobenchx_merged, geobenchx_scorer, geobenchx_steps
from agentic_evals.tasks.geobenchx.task import geobenchx

_install_openai_compat()
