#!/bin/bash
# Pod console for agentic-evals.
#
# One byobu session named "evals", and Inspect View on 127.0.0.1:7575.
# The viewer stays on loopback: Inspect rejects a browser Origin of https
# when the server itself is plain HTTP behind Runpod's proxy.
#
# On the pod:
#   scripts/remote_console.sh
#
# Attach from a laptop (this is the terminal):
#   ssh -t -p 32580 -i ~/.ssh/id_ed25519 root@HOST \
#     'cd /workspace/agentic-evals && scripts/remote_console.sh attach'
#
# Dashboard, in another local terminal, then open http://127.0.0.1:7575 :
#   ssh -L 7575:127.0.0.1:7575 -p 32580 -i ~/.ssh/id_ed25519 root@HOST

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:/workspace/gdal/bin:${PATH}"
export LD_LIBRARY_PATH="/workspace/gdal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_LINK_MODE=copy
SESSION=evals

if ! command -v byobu >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y byobu
fi

ensure_session() {
  if ! byobu has-session -t "$SESSION" 2>/dev/null; then
    byobu new-session -d -s "$SESSION" -n log -c "$ROOT"
    byobu send-keys -t "$SESSION:log" \
      "tail -n 50 -f logs/full-llama4-maverick-ap.log" C-m
  fi
  if ! byobu list-windows -t "$SESSION" -F '#{window_name}' | grep -qx view; then
    byobu new-window -d -t "$SESSION" -n view -c "$ROOT"
    byobu send-keys -t "$SESSION:view" \
      "export PATH=\"\$HOME/.local/bin:/workspace/gdal/bin:\$PATH\"; export LD_LIBRARY_PATH=/workspace/gdal/lib; uv run inspect view --log-dir logs --host 127.0.0.1 --port 7575" C-m
  fi
}

case "${1:-ensure}" in
  attach)
    ensure_session
    exec byobu attach -t "$SESSION"
    ;;
  ensure)
    ensure_session
    echo "session ${SESSION} ready (windows: log, view)"
    echo "viewer http://127.0.0.1:7575 via ssh -L 7575:127.0.0.1:7575"
    ;;
  *)
    echo "usage: $0 [ensure|attach]" >&2
    exit 2
    ;;
esac
