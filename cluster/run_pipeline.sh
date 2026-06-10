#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

RAW_DATA=${RAW_DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal.jsonl}
VISIBLE_DATA=${VISIBLE_DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal_visible_pipes_v3_death_after_collision.jsonl}
OUT=${OUT:-/scratch/gilbreth/mangla/flappybird/checkpoints/tokenized_flat_lm_terminal_collision_v3_50k_death_w10_50ksteps}
STEPS=${STEPS:-50000}

collect_id=$(sbatch --parsable cluster/collect_data.sbatch)
preprocess_id=$(sbatch --parsable --dependency=afterok:"$collect_id" \
  --export=ALL,RAW_DATA="$RAW_DATA",VISIBLE_DATA="$VISIBLE_DATA" \
  cluster/preprocess_data.sbatch)
train_id=$(sbatch --parsable --dependency=afterok:"$preprocess_id" \
  --export=ALL,VISIBLE_DATA="$VISIBLE_DATA",OUT="$OUT",STEPS="$STEPS",DEATH_WEIGHT=10,DONE_POSITIVE_WEIGHT=10 \
  cluster/train_model.sbatch)

echo "collect_job=$collect_id"
echo "preprocess_job=$preprocess_id (after collect)"
echo "train_job=$train_id (after preprocess, steps=$STEPS)"
echo "raw_data=$RAW_DATA"
echo "visible_data=$VISIBLE_DATA"
echo "checkpoint_dir=$OUT"
