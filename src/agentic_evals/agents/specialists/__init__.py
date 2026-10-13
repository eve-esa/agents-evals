"""Multi-agent solver: an orchestrator over tool-specialist sub-agents, each served over A2A.

- ``clustering``: splits a benchmark's tools into specialists, by embedding
  clusters (Tool-Planner) or one per MCP server file. Runs offline; the result
  is committed under ``clusters/`` and read at eval time.
- ``describer``: has a chat model write each specialist's AgentCard text,
  AgentSkills and role prompt.
- ``a2a``: serves each specialist as an A2A server on localhost, and gives the
  orchestrator one ``ask_<slug>`` tool per agent card.
- ``graph``: the Inspect solver, ``langgraph_specialists``.
"""

from agentic_evals.agents.specialists.graph import langgraph_specialists

__all__ = ["langgraph_specialists"]
