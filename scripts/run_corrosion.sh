#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/corrosion_app.py --server.port "${PORT:-8501}"
