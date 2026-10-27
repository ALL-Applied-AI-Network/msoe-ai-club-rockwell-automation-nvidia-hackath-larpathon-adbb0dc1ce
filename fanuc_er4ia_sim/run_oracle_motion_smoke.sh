#!/usr/bin/env bash
set -euo pipefail

# Phase 3A is intentionally split across two independent Isaac processes:
# 1. capture_scene.py starts Isaac, recreates Dataset 3, writes the synchronized
#    1920x1080 before image plus scene_state.json/ground_truth.json, then exits.
# 2. oracle_motion_smoke.py is a host-only step that reads that captured state,
#    labels SortCube2's pose oracle_simulation_pose, and writes the one-pick plan.
# 3. sort_scene.py starts a fresh Isaac process, recreates all ten objects, and
#    validates SortCube2's current pose/inventory against the plan before its
#    first articulation command. It then executes exactly one pick-and-place.
# 4. Required motion evidence is validated before optional video processing.
#    Video status is recorded separately and cannot invalidate passed motion.

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaac-sim}"
RUN="${1:?usage: run_oracle_motion_smoke.sh RUN_DIR}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Run directory must be empty: $RUN" >&2
    exit 1
fi

"$ISAAC/python.sh" "$HERE/capture_scene.py" \
    --out "$RUN" --camera table --resolution 1920x1080
cp "$RUN/world_camera.png" "$RUN/before_table_camera.png"
"$ISAAC/python.sh" "$HERE/oracle_motion_smoke.py" --run "$RUN"
"$ISAAC/python.sh" "$HERE/sort_scene.py" --run "$RUN" --camera table
"$ISAAC/python.sh" "$HERE/oracle_smoke_postprocess.py" \
    validate-motion --run "$RUN"

VIDEO_STATUS="unavailable"
VIDEO_ERROR="optional encoder dependency is unavailable"
VIDEO_OUTPUT="side_execution.mp4"
if DEPENDENCY_STATUS=$("$ISAAC/python.sh" "$HERE/encode_video.py" \
    --check-dependencies 2>&1); then
    echo "$DEPENDENCY_STATUS"
    if "$ISAAC/python.sh" "$HERE/encode_video.py" \
        --frames "$RUN/side_frames" --out "$RUN/$VIDEO_OUTPUT" --fps 30 \
        > >(tee "$RUN/video_encoding.log") 2>&1; then
        if [ -s "$RUN/$VIDEO_OUTPUT" ]; then
            VIDEO_STATUS="success"
            VIDEO_ERROR=""
        else
            VIDEO_STATUS="failed"
            VIDEO_ERROR="encode_video.py exited successfully but produced no video"
            echo "[oracle-smoke] WARNING: $VIDEO_ERROR" >&2
        fi
    else
        ENCODE_EXIT=$?
        VIDEO_STATUS="failed"
        VIDEO_ERROR="encode_video.py exited with status $ENCODE_EXIT; see video_encoding.log"
        echo "[oracle-smoke] WARNING: $VIDEO_ERROR" >&2
    fi
else
    VIDEO_ERROR="$DEPENDENCY_STATUS"
    echo "[oracle-smoke] WARNING: $VIDEO_ERROR; video not generated" >&2
fi
"$ISAAC/python.sh" "$HERE/oracle_smoke_postprocess.py" record-video \
    --run "$RUN" --status "$VIDEO_STATUS" --error "$VIDEO_ERROR" \
    --output "$VIDEO_OUTPUT"

echo "[oracle-smoke] PASS: one red cube placed in Red Slot 0; video=$VIDEO_STATUS"
