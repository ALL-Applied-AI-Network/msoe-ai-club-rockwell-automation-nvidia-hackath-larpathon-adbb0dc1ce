#!/usr/bin/env bash
set -euo pipefail

# Two-seed infrastructure gate.  This deliberately exercises two sequential
# USD stages in one SimulationApp before any full twelve-seed rerun is allowed.
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
RUN="${1:?usage: run_phase4b_lifecycle_smoke.sh RUN_DIR}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Run directory must be empty: $RUN" >&2
    exit 1
fi

"$ISAAC/python.sh" "$HERE/phase4b_capture.py" --out "$RUN" --seeds 4100 4104
"$ISAAC/python.sh" "$HERE/phase4b_process.py" process --run "$RUN"
"$ISAAC/python.sh" "$HERE/phase4b_process.py" validate-output \
    --run "$RUN" --require-smoke-go

echo "[phase4b-smoke] GO: seeds 4100 and 4104 passed lifecycle validation"
