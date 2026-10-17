#!/usr/bin/env python3
"""
Make the three rho_e caps in spectroxide's solver.rs overridable (raise-only) through the environment variable
SPECTROXIDE_RHO_E_CAP.  Without the variable the behaviour is IDENTICAL to the original (1.5 / 3.0 / 2.0).

    python3 patch_rho_cap.py spectroxide-src/src/solver.rs            # apply (writes solver.rs.bak first)
    python3 patch_rho_cap.py spectroxide-src/src/solver.rs --revert   # restore the backup
    cd spectroxide-src && cargo build --release                       # REQUIRED: the Python wrapper does not rebuild

The patch refuses to run unless each target line occurs exactly once, so it cannot half-apply.
"""
import shutil, sys
from pathlib import Path

EDITS = [
    ("let rho_e_new = rho_e_raw.clamp(0.0, 1.5);",
     "let rho_e_new = rho_e_raw.clamp(0.0, rho_e_cap().unwrap_or(1.5));"),
    ("let rho_clamped = rho_e_out.clamp(0.0, 3.0);",
     "let rho_clamped = rho_e_out.clamp(0.0, rho_e_cap().unwrap_or(3.0));"),
    ("(rho_e - inj_contribution).clamp(0.5, 2.0)",
     "(rho_e - inj_contribution).clamp(0.5, rho_e_cap().unwrap_or(2.0))"),
]
HELPER = '''
// ---- added by patch_rho_cap.py ------------------------------------------------------------------
// SPECTROXIDE_RHO_E_CAP can only RAISE the rho_e caps (values < 2 or non-finite are ignored).
fn rho_e_cap() -> Option<f64> {
    static CAP: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("SPECTROXIDE_RHO_E_CAP")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 2.0)
    })
}
// ---- end of patch -------------------------------------------------------------------------------
'''

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1]); bak = path.with_suffix(".rs.bak")
    if "--revert" in sys.argv:
        if not bak.exists():
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, path); print(f"restored {path} from {bak}"); return
    text = path.read_text()
    if "fn rho_e_cap()" in text:
        sys.exit("already patched (found fn rho_e_cap); nothing to do")
    for old, _ in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"ABORT: expected exactly 1 occurrence of  {old!r}  but found {n}.  Nothing was changed.")
    shutil.copy2(path, bak)
    for old, new in EDITS:
        text = text.replace(old, new)
    path.write_text(text.rstrip("\n") + "\n" + HELPER)
    print(f"patched {path}  (backup: {bak})")
    print("next:  cd spectroxide-src && cargo build --release")

if __name__ == "__main__":
    main()
