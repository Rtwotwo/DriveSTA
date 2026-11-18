#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
cd "$DRIVESTA_ROOT"
dataset="${1:?Usage: mine.sh navtrainval|simscale estimate|manifest|run|status|backfill [arguments]}"
action="${2:?Specify an action}"
shift 2
output="${MINING_OUTPUT:-$DRIVESTA_ROOT/data/mining/$dataset}"
case "$dataset" in
  navtrainval)
    exec "$PYTHON_BIN" feedback/scripts/prepare_scene_dataset.py "$action" \
      --config feedback/configs/high_score_256.yaml --output "$output" \
      --workers "${MINING_WORKERS:-8}" "$@" ;;
  simscale)
    if [[ "$action" == status ]]; then
      exec "$PYTHON_BIN" -m json.tool "$output/status.json"
    fi
    extra=()
    case "$action" in
      manifest) extra+=(--manifest-only) ;;
      run) ;;
      *) echo 'SimScale actions: manifest, run, status' >&2; exit 2 ;;
    esac
    exec "$PYTHON_BIN" feedback/scripts/prepare_simscale_dataset.py \
      --config feedback/configs/simscale_adaptive.yaml --output "$output" \
      --workers "${MINING_WORKERS:-8}" "${extra[@]}" "$@" ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
