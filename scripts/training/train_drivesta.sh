#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
cd "$DRIVESTA_ROOT"
variant="${1:?Usage: train.sh base_proposal|base|scale_proposal|scale [Hydra overrides]}"
shift
case "$variant" in
  base_proposal|base|scale_proposal|scale) ;;
  *) echo "Unknown DriveSTA profile: $variant" >&2; exit 2 ;;
esac
overrides=()
if [[ -n "${RANK_WEIGHT:-}" ]]; then
  overrides+=("agent.loss.rank_weight=$RANK_WEIGHT")
fi
exec "$PYTHON_BIN" -m navsim.planning.script.run_training \
  "--config-name=drivesta_$variant" "${overrides[@]}" "$@"
