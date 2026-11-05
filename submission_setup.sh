#!/usr/bin/env bash
# Network is used for dependencies/upstream only; no model API is called.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
py="${PYTHON_BIN:-python3}"
"$py" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required; Python 3.12 recommended"'
if [ ! -d lits-llm ]; then
  git clone https://github.com/xinzhel/lits-llm.git lits-llm
  git -C lits-llm checkout 1ef347bee7b3f2c10dc504342589b8c951794c2d
fi
test "$(git -C lits-llm rev-parse HEAD)" = '1ef347bee7b3f2c10dc504342589b8c951794c2d'
if [ ! -d .venv ]; then "$py" -m venv .venv; fi
.venv/bin/python -m pip install 'setuptools>=61.0' wheel
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e lits-llm
# Preserve the already supplied frozen splits. Do not regenerate them silently.
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0,'scripts')
from common import verify_manifest
verify_manifest()
print('Pinned upstream and supplied frozen task files verified.')
PY
.venv/bin/python scripts/check_api_guard.py
.venv/bin/python scripts/check_search_fixes.py
.venv/bin/python scripts/check_cycle_pruning.py
.venv/bin/python scripts/check_low_score.py
