#!/usr/bin/env bash
# One-shot environment setup for cmb_scan.py
#   - creates a Python venv (.venv)
#   - installs Rust (rustup) if missing
#   - clones + builds + installs spectroxide
#   - installs the other Python packages
#
# Usage:   bash setup.sh
# Options: VENV=path  SPECTROXIDE_DIR=path  bash setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
SPX_DIR="${SPECTROXIDE_DIR:-$HERE/spectroxide}"
SPX_URL="https://github.com/bakerem/spectroxide.git"

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

# ── 5. spectroxide: clone, build, install ──────────────────────────────────
say "spectroxide"
if [ -d "$SPX_DIR/.git" ]; then
  git -C "$SPX_DIR" pull --ff-only || echo "(pull failed, using existing checkout)"
else
  git clone "$SPX_URL" "$SPX_DIR"
fi
cd "$SPX_DIR"
if [ -x ./install.sh ]; then
  ./install.sh            # installs into the active (venv) Python
else
  cargo build --release
  python -m pip install -e "python/.[plot]"
fi
cd "$HERE"

# ── 6. verify ──────────────────────────────────────────────────────────────
say "Verifying"
python - <<'EOF'
import spectroxide
from spectroxide.solver import solve
print("spectroxide OK:", getattr(spectroxide, "__file__", "?"))
EOF

if [ -d "$HERE/Cascades" ]; then
  python -c "import sys; sys.path.insert(0, '$HERE'); from Cascades.cascade.mckeen_bridge import make_mckeen_dq_dz; print('Cascades OK')" \
    || echo "WARNING: Cascades found but import failed (missing dependency?)."
else
  echo "WARNING: no Cascades/ folder next to setup.sh. Copy it here before running cmb_scan.py."
fi

say "Done"
echo "Activate with:   source \"$VENV/bin/activate\""
echo "Then run:        python cmb_scan.py --list"
