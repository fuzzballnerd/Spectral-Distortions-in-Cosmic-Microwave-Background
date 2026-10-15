#!/usr/bin/env bash
# One-shot, repeatable setup for cmb_scan.py on a fresh machine.
#
#   bash setup.sh
#
# Expects, next to this script:   cmb_scan.py  run_scan.sh  Cascades/  spectroxide.zip
# Does:  tmux  ->  Rust  ->  venv  ->  numpy/scipy/...  ->  YOUR spectroxide.zip (never GitHub)  ->  self-test
# Options (env):  VENV=path  SPECTROXIDE_ZIP=path  FORCE=1 (re-extract/re-install)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
SPX_DIR="$HERE/spectroxide-src"            # must NOT be called "spectroxide": that would shadow the real package
SPX_ZIP="${SPECTROXIDE_ZIP:-$HERE/spectroxide.zip}"

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. sanity on the bundle ────────────────────────────────────────────────
say "Checking bundle"
for f in cmb_scan.py run_scan.sh; do [ -f "$HERE/$f" ] || die "$f is missing next to setup.sh"; done
[ -d "$HERE/Cascades" ] || die "Cascades/ folder is missing next to setup.sh"
[ -f "$SPX_ZIP" ]       || die "spectroxide.zip not found at $SPX_ZIP"
# a folder literally called 'spectroxide' next to cmb_scan.py hides the installed package -> move it away
if [ -e "$HERE/spectroxide" ]; then
  mv "$HERE/spectroxide" "$HERE/spectroxide.moved-away.$(date +%s)"
  echo "Moved a folder named 'spectroxide' out of the way (it would shadow the installed package)."
fi
# optional integrity check: put the sha256 of your known-good zip in spectroxide.zip.sha256
ZIP_SHA="$( (sha256sum "$SPX_ZIP" 2>/dev/null || shasum -a 256 "$SPX_ZIP") | cut -d' ' -f1)"
echo "spectroxide.zip sha256: $ZIP_SHA"
if [ -f "$SPX_ZIP.sha256" ]; then
  [ "$(cut -d' ' -f1 "$SPX_ZIP.sha256")" = "$ZIP_SHA" ] || die "spectroxide.zip does not match spectroxide.zip.sha256 (corrupt or different version)"
  echo "zip checksum matches the reference"
fi
[ -z "${CONDA_DEFAULT_ENV:-}" ] || echo "Note: conda env '$CONDA_DEFAULT_ENV' is active; the venv below takes priority inside this script."

# ── 1. system prerequisites ────────────────────────────────────────────────
say "Prerequisites"
for t in curl python3; do command -v "$t" >/dev/null || die "Missing '$t' (e.g. sudo apt install curl python3 python3-venv build-essential)"; done
command -v cc >/dev/null || echo "WARNING: no C compiler; Rust may fail to link (sudo apt install build-essential)."
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python >= 3.9 required"'
python3 -c 'import venv, ensurepip' 2>/dev/null || die "python3-venv missing (sudo apt install python3-venv)"

# ── 1b. tmux ───────────────────────────────────────────────────────────────
say "tmux"
if command -v tmux >/dev/null; then tmux -V; else
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
  if   command -v apt-get >/dev/null; then $SUDO apt-get update && $SUDO apt-get install -y tmux
  elif command -v dnf     >/dev/null; then $SUDO dnf install -y tmux
  elif command -v yum     >/dev/null; then $SUDO yum install -y tmux
  elif command -v pacman  >/dev/null; then $SUDO pacman -S --noconfirm tmux
  elif command -v brew    >/dev/null; then brew install tmux
  else echo "WARNING: could not install tmux automatically; install it manually."; fi
fi

# ── 2. Rust ────────────────────────────────────────────────────────────────
say "Rust toolchain"
if ! command -v cargo >/dev/null && [ ! -f "$HOME/.cargo/env" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
cargo --version

# ── 3. venv (always use the venv's python, whatever else is active) ────────
say "Python venv: $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python() { "$VENV/bin/python" "$@"; }       # immune to an active conda 'base' env
python -m pip install --upgrade pip wheel setuptools

# ── 4. other Python packages ───────────────────────────────────────────────
say "Python packages"
python -m pip install numpy scipy matplotlib pandas tqdm
# add anything else your Cascades package needs here

# ── 5. spectroxide from YOUR zip ───────────────────────────────────────────
say "spectroxide (from $(basename "$SPX_ZIP"))"
if [ "${FORCE:-0}" = "1" ] || [ ! -f "$SPX_DIR/.from_zip_sha256" ] || [ "$(cat "$SPX_DIR/.from_zip_sha256")" != "$ZIP_SHA" ]; then
  rm -rf "$SPX_DIR"
  TMPX="$(mktemp -d)"
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$SPX_ZIP" "$TMPX"
  rm -rf "$TMPX/__MACOSX"
  mapfile -t TOP < <(ls -A "$TMPX")
  if [ "${#TOP[@]}" -eq 1 ] && [ -d "$TMPX/${TOP[0]}" ]; then mv "$TMPX/${TOP[0]}" "$SPX_DIR"; else mv "$TMPX" "$SPX_DIR"; fi
  rm -rf "$TMPX"
  echo "$ZIP_SHA" > "$SPX_DIR/.from_zip_sha256"
  NEED_INSTALL=1
else
  echo "Using $SPX_DIR (already extracted from this exact zip)"
  NEED_INSTALL=0
fi
# (re)install if freshly extracted, forced, or the installed package isn't this tree
if [ "$NEED_INSTALL" = "1" ] || ! python -c "import spectroxide.solver as m,sys; sys.exit(0 if m.__file__.startswith('$SPX_DIR') else 1)" 2>/dev/null; then
  python -m pip uninstall -y spectroxide >/dev/null 2>&1 || true
  ( cd "$SPX_DIR"
    if [ -f install.sh ]; then bash install.sh
    else
      cargo build --release
      if   [ -f python/pyproject.toml ]; then python -m pip install -e "python/.[plot]"
      elif [ -f pyproject.toml ];        then python -m pip install -e ".[plot]"
      else die "no install.sh or pyproject.toml in $SPX_DIR"; fi
    fi )
fi

# ── 6. self-test ───────────────────────────────────────────────────────────
say "Self-test"
( cd "$HERE" && SPX_DIR="$SPX_DIR" python - <<'EOF2'
import os, sys, inspect
import spectroxide, spectroxide.solver as m
spx = os.environ["SPX_DIR"]
assert getattr(spectroxide, "__file__", None), "import spectroxide resolved to a bare folder (shadowing)"
assert hasattr(spectroxide, "solve"), "spectroxide.solve missing"
assert os.path.realpath(m.__file__).startswith(os.path.realpath(spx)), \
    f"installed spectroxide is NOT the one from your zip: {m.__file__}"
src = inspect.getsource(m)
assert "belong to method='table'" not in src, \
    "this spectroxide rejects n_x/x_max/x_min for method='pde' (wrong/newer version than your notebook's)"
print("spectroxide OK, from your zip:", m.__file__)
EOF2
)
( cd "$HERE" && python cmb_scan.py --list )     # imports Cascades + spectroxide exactly as the real run does

say "Done"
echo "Start a scan (detached, survives disconnects):   bash run_scan.sh 10 15 25"
echo "Check on it:                                      tmux attach -t cmb      (detach: Ctrl+b, d)"
