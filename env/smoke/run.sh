#!/usr/bin/env bash
# CPU-only smoke test of the TreeOPSD math. Needs python3 with numpy, torch, pytest.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pytest -q env/smoke "$@"
