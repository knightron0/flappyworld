#!/bin/bash
#SBATCH --job-name=flappy-flatlm-base-rope
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G

# Baseline flat LM with per-episode global monotonic positions + RoPE.
# Full gap loss on every frame (no gap_copy training regime).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs /scratch/gilbreth/mangla/flappybird/checkpoints
source /scratch/gilbreth/mangla/venvs/flappyworld/bin/activate

VISIBLE_DATA=${VISIBLE_DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal_visible_pipes_v3_death_after_collision.jsonl}
OUT=${OUT:-/scratch/gilbreth/mangla/flappybird/checkpoints/tokenized_flat_lm_terminal_collision_v3_50k_death_w10_global_rope_baseline}
STEPS=${STEPS:-50000}

echo "gap_supervision=all_frames position_scheme=global_rope play_gap_copy=no"

python -m scripts.train_model \
  --data "$VISIBLE_DATA" \
  --data-format visible_tokens \
  --out-dir "$OUT" \
  --steps "$STEPS" \
  --eval-interval 1000 \
  --eval-iters 25 \
  --batch-size 64 \
  --block-size 640 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 192 \
  --dropout 0.1 \
  --weight-decay 0.01 \
  --death-weight 10 \
  --done-positive-weight 10 \
  --max-position 65536

echo "checkpoint_dir=$OUT"
