#!/usr/bin/env bash
set -euo pipefail
cd /home/chenrui/projects/MiniLLM
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/minillm-mpl
export PYTHONUNBUFFERED=1
exec /home/chenrui/.venvs/minillm/bin/python -u run_overnight.py
