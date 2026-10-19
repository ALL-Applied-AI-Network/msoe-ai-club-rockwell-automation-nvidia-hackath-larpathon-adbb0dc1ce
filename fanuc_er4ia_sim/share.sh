#!/usr/bin/env bash
# Build the curated participant ZIP. See package_hackathon.py --help.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
exec "$HOST_PYTHON" "$HERE/package_hackathon.py" "$@"
