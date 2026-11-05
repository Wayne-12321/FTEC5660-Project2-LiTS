#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required; set PYTHON_BIN"'
if [ ! -d lits-llm ]; then
  git clone https://github.com/xinzhel/lits-llm.git lits-llm
  git -C lits-llm checkout 1ef347bee7b3f2c10dc504342589b8c951794c2d
fi
test "$(git -C lits-llm rev-parse HEAD)" = "1ef347bee7b3f2c10dc504342589b8c951794c2d" || {
  echo 'Unexpected upstream revision; preserve local work and review manually.' >&2
  exit 1
}
if [ ! -d .venv ]; then "$python_bin" -m venv .venv; fi
if [ -f requirements-lock.txt ]; then
  .venv/bin/python -m pip install -r requirements-lock.txt
else
  .venv/bin/python -m pip install -r requirements-api.txt
fi
.venv/bin/python -m pip install --no-deps --no-build-isolation -e lits-llm
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/check_api_guard.py
.venv/bin/python scripts/check_search_fixes.py
.venv/bin/python scripts/check_cycle_pruning.py
.venv/bin/python scripts/check_plan_verify.py
.venv/bin/python scripts/check_local_repair.py
