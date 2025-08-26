#!/usr/bin/env bash
set -euo pipefail
# ========== USER CONFIG ==========
OVITOS="$HOME/Desktop/ovito-pro-3.12.4-x86_64/bin/ovitos"
PY_FILTER="02_filter_dump_ovito.py"
# =================================

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 IN_DUMP OUT_DUMP START_FRAME END_FRAME [--types \"1,2\" ...]" >&2
  exit 1
fi
IN="$1"; OUT="$2"; START="$3"; END="$4"; shift 4
exec "$OVITOS" "$PY_FILTER" --in "$IN" --out "$OUT" --start "$START" --end "$END" "$@"
