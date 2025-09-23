#!/usr/bin/env bash
set -e
export PYTHONPATH=/workspace
# bind to 0.0.0.0 for RunPod
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}
