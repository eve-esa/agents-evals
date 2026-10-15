#!/usr/bin/env bash
# SageMaker lifecycle hook (OnStart) for the workshop notebook instance. Safe to run again.
# SageMaker stops a lifecycle script after 5 minutes and the install takes longer, so setup.sh runs
# in the background. Progress goes to ~/bifold-setup.log.
#
# FOLDER overrides the folder to set up (default: the folder this script sits in).
set -euo pipefail

FOLDER="${FOLDER:-$(cd "$(dirname "$0")" && pwd)}"
LOG="$HOME/bifold-setup.log"

if [[ ! -f "$FOLDER/setup.sh" ]]; then
  echo "setup.sh not found in $FOLDER. Set FOLDER to the evaluating-eo-agents-bifold folder." >&2
  exit 1
fi

if "$FOLDER/.venv/bin/python" -c 'import agentic_evals, ipykernel' >/dev/null 2>&1; then
  echo "bifold environment already present in $FOLDER"
  exit 0
fi

nohup bash -c 'cd "$1" && bash setup.sh; echo "setup exit $?"' _ "$FOLDER" >>"$LOG" 2>&1 &
echo "bifold setup started in the background. Follow it with: tail -f $LOG"
