#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/minillm-mpl
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -u run_overnight.py
