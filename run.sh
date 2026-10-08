#!/usr/bin/env sh
set -eu

uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' INT TERM EXIT
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
