#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
cd "$DRIVESTA_ROOT"
variant="${1:?Usage: evaluate.sh base_proposal|base|scale_proposal|scale [Hydra overrides]}"
shift
case "$variant" in
  base_proposal|scale_proposal) config=drivesta_proposal_eval; agent=drivesta_proposal ;;
  base|scale) config=drivesta_eval; agent=drivesta ;;
  *) echo "Unknown DriveSTA profile: $variant" >&2; exit 2 ;;
esac
checkpoint="${CHECKPOINT_PATH:-$DRIVESTA_ROOT/weights/Checkpoints/$variant.ckpt}"
output="${OUTPUT_DIR:-$NAVSIM_EXP_ROOT/evaluation/${variant}_$(date +%Y%m%d_%H%M%S)}"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone \
  "--nproc-per-node=${GPUS_PER_NODE:-1}" --max-restarts=0 \
  --module navsim.planning.script.run_pdm_score_drivesta_multi_gpu \
  "--config-name=$config" "agent=$agent" \
  "agent.checkpoint_path='$checkpoint'" "training_config_path='$DRIVESTA_ROOT/navsim/planning/script/config/checkpoints/$variant.yaml'" \
  "output_dir='$output'" "experiment_name=drivesta_$variant" \
  "trainer.params.devices=${GPUS_PER_NODE:-1}" trainer.params.precision=32-true \
  "metric_cache_path='${METRIC_CACHE_PATH:-$NAVSIM_CACHE_ROOT/metric_cache}'" "$@"
