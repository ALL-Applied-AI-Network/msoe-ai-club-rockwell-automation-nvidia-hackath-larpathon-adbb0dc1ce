#!/usr/bin/env bash
# Capture -> Cosmos -> plan -> physical execution -> review videos.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
: "${COSMOS:?Set COSMOS to the Python interpreter in your Cosmos environment}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
for executable in "$ISAAC/python.sh" "$COSMOS" "$HOST_PYTHON"; do
    if [ ! -x "$executable" ]; then
        echo "Python launcher missing or not executable: $executable" >&2
        exit 1
    fi
done
RUN="${1:-$HERE/runs/cube_stack_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi
"$ISAAC/python.sh" "$HERE/capture_scene.py" --out "$RUN"
"$COSMOS" "$HERE/perceive.py" --run "$RUN" --stack-cubes
"$HOST_PYTHON" "$HERE/plan_sort.py" --run "$RUN"
"$ISAAC/python.sh" "$HERE/sort_scene.py" --run "$RUN"
# Isaac shutdown can return zero even after an exception; check the result.
"$HOST_PYTHON" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1]) / 'execution_report.json'
if not report.is_file():
    raise SystemExit('Execution report missing; inspect the simulator output.')
result = json.loads(report.read_text())
if not result.get('task_complete'):
    raise SystemExit(f"Stack incomplete: {result.get('abort_reason')}; inspect {report}")
PY
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/frames" --out "$RUN/sort_table.mp4" --fps 30
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/side_frames" --out "$RUN/sort_side.mp4" --fps 30
echo "Review $RUN: videos, detections_overlay.png, sort_plan.json, perception_evaluation.json, execution_report.json"
