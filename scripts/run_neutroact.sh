#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/neutroact_app.py --server.port "${PORT:-8501}"
