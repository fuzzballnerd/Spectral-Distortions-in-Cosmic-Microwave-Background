import os
import sys

# Ensure the repo root (this file's directory, which contains the `cascade`
# package) is importable regardless of how/where pytest is invoked from --
# `python3 test_cpN...py` directly, bare `pytest`, `pytest tests/`, or
# `python3 -m pytest` from any cwd. Each test file's own
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) already
# covers direct-script and most pytest invocations; this conftest.py is the
# belt-and-braces fix for the remaining cases (e.g. plain `pytest` run from
# a directory other than the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
