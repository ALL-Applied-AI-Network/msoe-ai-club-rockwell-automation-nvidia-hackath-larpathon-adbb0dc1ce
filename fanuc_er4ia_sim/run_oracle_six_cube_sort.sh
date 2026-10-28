#!/usr/bin/env bash
set -euo pipefail

# Phase 3B remains an isolated oracle-only physics validation:
# 1. Isaac captures Dataset 3 and exits.
# 2. A host-only planner writes the all-or-nothing six-cube oracle plan.
# 3. A fresh Isaac process validates all ten recreated objects, preflights all
#    six trajectories, and executes the shared pick/place segments with strict
#    per-cube safety gates. One decimated table-camera stream is retained.
# 4. Required motion evidence is validated before optional MP4 processing.

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
RUN="${1:?usage: run_oracle_six_cube_sort.sh RUN_DIR}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Run directory must be empty: $RUN" >&2
    exit 1
fi

"$ISAAC/python.sh" "$HERE/capture_scene.py" \
    --out "$RUN" --camera table --resolution 1920x1080
cp "$RUN/world_camera.png" "$RUN/before_table_camera.png"
"$ISAAC/python.sh" "$HERE/oracle_six_cube_sort.py" --run "$RUN"
"$ISAAC/python.sh" "$HERE/sort_scene.py" \
    --run "$RUN" --camera table --single-frame-stream --frame-save-stride 3
"$ISAAC/python.sh" "$HERE/oracle_smoke_postprocess.py" \
    validate-motion --run "$RUN"

VIDEO_STATUS="unavailable"
VIDEO_ERROR="optional encoder dependency is unavailable"
VIDEO_OUTPUT="table_execution.mp4"
if DEPENDENCY_STATUS=$("$ISAAC/python.sh" "$HERE/encode_video.py" \
    --check-dependencies 2>&1); then
    echo "$DEPENDENCY_STATUS"
    if "$ISAAC/python.sh" "$HERE/encode_video.py" \
        --frames "$RUN/frames" --out "$RUN/$VIDEO_OUTPUT" --fps 10 \
        > >(tee "$RUN/video_encoding.log") 2>&1; then
        if [ -s "$RUN/$VIDEO_OUTPUT" ]; then
            VIDEO_STATUS="success"
            VIDEO_ERROR=""
        else
            VIDEO_STATUS="failed"
            VIDEO_ERROR="encode_video.py exited successfully but produced no video"
            echo "[oracle-six] WARNING: $VIDEO_ERROR" >&2
        fi
    else
        ENCODE_EXIT=$?
        VIDEO_STATUS="failed"
        VIDEO_ERROR="encode_video.py exited with status $ENCODE_EXIT; see video_encoding.log"
        echo "[oracle-six] WARNING: $VIDEO_ERROR" >&2
    fi
else
    VIDEO_ERROR="$DEPENDENCY_STATUS"
    echo "[oracle-six] WARNING: $VIDEO_ERROR; video not generated" >&2
fi
"$ISAAC/python.sh" "$HERE/oracle_smoke_postprocess.py" record-video \
    --run "$RUN" --status "$VIDEO_STATUS" --error "$VIDEO_ERROR" \
    --output "$VIDEO_OUTPUT"

echo "[oracle-six] PASS: all six cubes placed in assigned defect slots; video=$VIDEO_STATUS"
