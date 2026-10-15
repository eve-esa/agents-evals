#!/usr/bin/env bash
# Create this folder's virtualenv, the workshop kernel, the agentic-evals library, and an empty .env. Run it from anywhere.
set -euo pipefail
# Write .env first, so the EVE settings can be filled in before the library install.
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Wrote .env from .env.example. Fill in EVE_API_KEY, EVE_BASE_URL, and MODEL_NAME."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing it with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name agentic-evals-workshop --display-name "agentic-evals workshop"
PYTHON=.venv/bin/python ./install_library.sh

echo "Kernel: agentic-evals workshop"
echo "Start Jupyter from this folder: .venv/bin/jupyter notebook eval_agents_handson.ipynb"
