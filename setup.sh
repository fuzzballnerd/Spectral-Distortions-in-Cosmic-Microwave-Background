#!/usr/bin/env bash
# One-shot environment setup for cmb_scan.py
#   - creates a Python venv (.venv)
#   - installs Rust (rustup) if missing
#   - extracts spectroxide.zip (next to this script), builds + installs it
#   - installs the other Python packages
#
# Usage:   bash setup.sh
# Options: VENV=path  SPECTROXIDE_DIR=path  SPECTROXIDE_ZIP=path  FORCE=1  bash setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
SPX_DIR="${SPECTROXIDE_DIR:-$HERE/spectroxide-src}"   # NOT named "spectroxide": a folder with that name next to cmb_scan.py shadows the real package

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# ── 1. system prerequisites ────────────────────────────────────────────────
say "Checking prerequisites"
for t in git curl python3; do
  command -v "$t" >/dev/null || { echo "Missing '$t'. Install it first (e.g. sudo apt install git curl python3 python3-venv build-essential)."; exit 1; }
done
command -v cc >/dev/null || echo "WARNING: no C compiler found; Rust may need one to link (sudo apt install build-essential / xcode-select --install)."
python3 - <<'EOF'
import sys
assert sys.version_info >= (3, 9), "Python >= 3.9 required"
EOF

# ── 1b. tmux (keeps the scan running after you disconnect) ─────────────────
say "tmux"
if command -v tmux >/dev/null; then
  tmux -V
else
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
  if   command -v apt-get >/dev/null; then $SUDO apt-get update && $SUDO apt-get install -y tmux
  elif command -v dnf     >/dev/null; then $SUDO dnf install -y tmux
  elif command -v yum     >/dev/null; then $SUDO yum install -y tmux
  elif command -v pacman  >/dev/null; then $SUDO pacman -S --noconfirm tmux
  elif command -v brew    >/dev/null; then brew install tmux
  else echo "WARNING: could not install tmux automatically; install it manually."; fi
fi

# ── 2. Rust toolchain ──────────────────────────────────────────────────────
say "Rust toolchain"
if ! command -v cargo >/dev/null && [ ! -f "$HOME/.cargo/env" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
cargo --version

# ── 3. Python venv ─────────────────────────────────────────────────────────
say "Python virtual environment: $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ── 4. other Python dependencies ───────────────────────────────────────────
say "Python packages"
python -m pip install numpy scipy matplotlib
# If your Cascades package needs more (pandas, astropy, ...), add them here:
# python -m pip install pandas

# ── 5. spectroxide: extract YOUR zip, build, install ───────────────────────
say "spectroxide (from spectroxide.zip)"
SPX_ZIP="${SPECTROXIDE_ZIP:-$HERE/spectroxide.zip}"
if [ ! -d "$SPX_DIR" ] || [ "${FORCE:-0}" = "1" ]; then
  [ -f "$SPX_ZIP" ] || { echo "ERROR: $SPX_ZIP not found. Put spectroxide.zip next to setup.sh (or set SPECTROXIDE_ZIP=/path/to.zip)."; exit 1; }
  rm -rf "$SPX_DIR"
  TMPX="$(mktemp -d)"
  python - "$SPX_ZIP" "$TMPX" <<'EOF3'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
EOF3
  # drop macOS junk, and if the zip has a single top-level folder use that as the project root
  rm -rf "$TMPX/__MACOSX"
  mapfile -t TOP < <(ls -A "$TMPX")
  if [ "${#TOP[@]}" -eq 1 ] && [ -d "$TMPX/${TOP[0]}" ]; then mv "$TMPX/${TOP[0]}" "$SPX_DIR"; else mv "$TMPX" "$SPX_DIR"; fi
  rm -rf "$TMPX"
else
  echo "Using existing $SPX_DIR (set FORCE=1 to re-extract the zip)"
fi

python -m pip uninstall -y spectroxide >/dev/null 2>&1 || true     # drop any older / different install
cd "$SPX_DIR"
if [ -f install.sh ]; then
  bash install.sh          # installs into the active (venv) Python; run via bash since zips lose the exec bit
else
  cargo build --release
  if   [ -f python/pyproject.toml ]; then python -m pip install -e "python/.[plot]"
  elif [ -f pyproject.toml ];        then python -m pip install -e ".[plot]"
  else echo "ERROR: no install.sh or pyproject.toml found in $SPX_DIR"; exit 1; fi
fi
cd "$HERE"

# ── 6. verify ──────────────────────────────────────────────────────────────
say "Verifying"
( cd "$HERE" && python - <<'EOF2'
import spectroxide
from spectroxide.solver import solve
assert getattr(spectroxide, "__file__", None), (
    "'import spectroxide' resolved to a plain folder (namespace package) instead of the installed package. "
    "Rename/remove any folder called 'spectroxide' next to cmb_scan.py.")
assert hasattr(spectroxide, "solve"), "spectroxide.solve missing - the checked-out version does not match what cmb_scan.py expects"
print("spectroxide OK:", spectroxide.__file__)
EOF2
)

if [ -d "$HERE/Cascades" ]; then
  python -c "import sys; sys.path.insert(0, '$HERE'); from Cascades.cascade.mckeen_bridge import make_mckeen_dq_dz; print('Cascades OK')" \
    || echo "WARNING: Cascades found but import failed (missing dependency?)."
else
  echo "WARNING: no Cascades/ folder next to setup.sh. Copy it here before running cmb_scan.py."
fi

say "Done"
echo "Activate with:   source \"$VENV/bin/activate\""
echo "Then run:        python cmb_scan.py --list"
echo
echo "To run detached (survives disconnects):"
echo "   bash run_scan.sh 0.5 1 3 6        # starts tmux session 'cmb' with --dm 0.5 1 3 6"
echo "   tmux attach -t cmb                # re-attach later (detach again: Ctrl+b then d)"
