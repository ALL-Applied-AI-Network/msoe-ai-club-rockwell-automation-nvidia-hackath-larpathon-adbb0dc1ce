#!/usr/bin/env bash
set -euo pipefail

# Phase 4B is perception-only. One SimulationApp captures all twelve fresh
# stages; no articulation/controller path is invoked. Detector inference is
# completed for the full batch before evaluator-only truth is opened.
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
RUN="${1:?usage: run_phase4b_randomized_perception.sh RUN_DIR}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Run directory must be empty: $RUN" >&2
    exit 1
fi

"$ISAAC/python.sh" "$HERE/phase4b_capture.py" --out "$RUN"
"$ISAAC/python.sh" "$HERE/phase4b_process.py" process --run "$RUN"
"$ISAAC/python.sh" "$HERE/phase4b_process.py" validate-output --run "$RUN"

echo "[phase4b] complete; inspect aggregate_evaluation.json for GO/NO-GO"
