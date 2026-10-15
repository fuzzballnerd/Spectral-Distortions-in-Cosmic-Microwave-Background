#!/usr/bin/env bash
# Launch cmb_scan.py detached in tmux, shown to observers as "batman".
#
# Usage:   bash run_scan.sh 0.5 1 3 6          # Delta_m values for this machine
# Env:     SESSION=batman   EXTRA="--workers 8"
#
# Re-attach:  tmux attach -t batman     Detach: Ctrl+b then d
# Resuming is automatic: re-running the same command skips points already on disk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-batman}"
EXTRA="${EXTRA:-}"

[ $# -ge 1 ] || { echo "Usage: bash run_scan.sh <dm values...>"; exit 1; }
command -v tmux >/dev/null || { echo "tmux not installed; run setup.sh first."; exit 1; }
[ -f "$HERE/.venv/bin/activate" ] || { echo "No .venv found; run setup.sh first."; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

# The script is run through a symlink so `ps`/`top` show "batman.py", not "cmb_scan.py".
ln -sf "$HERE/cmb_scan.py" "$HERE/batman.py"

# Inner runner (its name and contents are what the tmux pane's command line shows).
RUNNER="$HERE/.batman_run.sh"
cat > "$RUNNER" <<INNER
#!/usr/bin/env bash
cd "$HERE"
source .venv/bin/activate
[ -f "\$HOME/.cargo/env" ] && source "\$HOME/.cargo/env"
python batman.py --dm "\$@" $EXTRA 2>&1 | tee -a batman.log
echo; echo "[ended - press Enter to close]"; read
INNER

tmux new-session -d -s "$SESSION" -n "$SESSION" "bash '$RUNNER' $*"
echo "Started tmux session '$SESSION' (log: $HERE/batman.log)"
echo "Attach:  tmux attach -t $SESSION"
