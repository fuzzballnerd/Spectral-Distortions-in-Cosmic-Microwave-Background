#!/usr/bin/env bash
# Launch cmb_scan.py inside a detached tmux session so it survives disconnects.
#
# Usage:   bash run_scan.sh 0.5 1 3 6          # the Delta_m values for this machine
# Env:     SESSION=cmb  (tmux session name)   EXTRA="--workers 8"  (extra cmb_scan.py args)
#
# Re-attach:  tmux attach -t cmb      Detach: Ctrl+b then d      Stop: Ctrl+c inside the session
# Resuming is automatic: re-running the same command skips points already on disk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-cmb}"
EXTRA="${EXTRA:-}"

[ $# -ge 1 ] || { echo "Usage: bash run_scan.sh <dm values...>   e.g. bash run_scan.sh 0.5 1 3 6"; exit 1; }
command -v tmux >/dev/null || { echo "tmux not installed; run setup.sh first."; exit 1; }
[ -f "$HERE/.venv/bin/activate" ] || { echo "No .venv found; run setup.sh first."; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (scan probably running). Attach with: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$HERE' && source .venv/bin/activate && [ -f \"\$HOME/.cargo/env\" ] && source \"\$HOME/.cargo/env\"; \
python cmb_scan.py --dm $* $EXTRA 2>&1 | tee -a scan_${SESSION}.log; \
echo; echo '[scan process ended - press Enter to close]'; read"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""
echo "Started tmux session '$SESSION' (log: $HERE/scan_${SESSION}.log)"
echo "Attach:  tmux attach -t $SESSION"
