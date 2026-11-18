#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
cd "$DRIVESTA_ROOT"
dataset="${1:?Usage: mine_epdms.sh navtrainval|simscale run|recover|status [arguments]}"
action="${2:?Specify an action}"
shift 2
output="${EPDMS_OUTPUT:-$DRIVESTA_ROOT/data/mining/${dataset}_epdms}"
case "$dataset:$action" in
  navtrainval:run)
    exec "$PYTHON_BIN" feedback/scripts/generate_epdms.py --out "$output" \
      --workers "${MINING_WORKERS:-8}" "$@" ;;
  navtrainval:recover)
    exec "$PYTHON_BIN" feedback/scripts/recover_navsim_epdms.py --root "$output" \
      --workers "${MINING_WORKERS:-8}" "$@" ;;
  simscale:run)
    exec "$PYTHON_BIN" feedback/scripts/generate_simscale_epdms.py --out "$output" \
      --workers "${MINING_WORKERS:-8}" "$@" ;;
  navtrainval:status|simscale:status)
    exec "$PYTHON_BIN" -m json.tool "$output/status.json" ;;
  *) echo "Unsupported EPDMS dataset/action: $dataset $action" >&2; exit 2 ;;
esac
