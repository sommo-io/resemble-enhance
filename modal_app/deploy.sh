#!/usr/bin/env bash
# Deploy the Modal app, then pre-build its memory snapshots (see warmup.py).
# Run from anywhere; needs the modal CLI (logged in) and uv.
set -euo pipefail
cd "$(dirname "$0")/.."

modal deploy modal_app/app.py
uv run --quiet --no-project --with modal python modal_app/warmup.py "resemble-enhance" "${WARMUP_CONTAINERS:-5}"
