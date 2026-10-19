#!/usr/bin/env bash
# Prepare only the host environment and locally generated materials.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python make_textures.py
.venv/bin/python -m unittest test_sort_planning test_cube_perception test_cube_pose test_runtime_config
echo "Host setup complete. Configure ISAAC and COSMOS next; see SETUP.md."
