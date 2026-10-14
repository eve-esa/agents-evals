#!/usr/bin/env bash
# Create this folder's virtualenv and an empty .env. Run it from anywhere.
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
else
  PY=python3
fi
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' \
  || { echo "Python 3.12+ is required (found $("$PY" --version 2>&1))" >&2; exit 1; }

"$PY" -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name agentic-evals-workshop --display-name "agentic-evals workshop"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Wrote .env from .env.example. Fill in EVE_API_KEY, EVE_BASE_URL, and MODEL_NAME."
fi

echo "Kernel: agentic-evals workshop"
echo "Start Jupyter from this folder: .venv/bin/jupyter notebook eval_agents_handson.ipynb"
