#!/usr/bin/env bash
# Install the agentic-evals library, with its Earth-Bench extra, into one Python environment.
# Non-interactive. Run it from anywhere, or let setup.sh and the notebook call it.
#
# PYTHON           interpreter to install into (default: this folder's .venv)
# AGENTS_EVALS_REF git ref to install from GitHub (default: feat/workshop-notebook)
# AGENTS_EVALS_URL repository URL (default: https://github.com/eve-esa/agents-evals.git)
#
# When this folder sits inside an agentic-evals checkout, that checkout is installed in editable mode.
# Otherwise the library is installed from GitHub at AGENTS_EVALS_REF.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
REF="${AGENTS_EVALS_REF:-feat/workshop-notebook}"
REPO_URL="${AGENTS_EVALS_URL:-https://github.com/eve-esa/agents-evals.git}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing it with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -f ../pyproject.toml ]] && grep -q '^name = "agentic-evals"' ../pyproject.toml; then
  echo "Installing agentic-evals from the checkout at $(cd .. && pwd), editable."
  SPEC_ARGS=(-e "..[earth-bench]")
else
  echo "Installing agentic-evals from ${REPO_URL} at ${REF}."
  SPEC_ARGS=("agentic-evals[earth-bench] @ git+${REPO_URL}@${REF}")
fi

uv pip install --python "$PYTHON" "${SPEC_ARGS[@]}"
"$PYTHON" -c 'import agentic_evals, importlib.metadata as m; print("agentic-evals", m.version("agentic-evals"), "importable")'
