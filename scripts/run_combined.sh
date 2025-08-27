#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/combined_app.py --server.port "${PORT:-8501}"
