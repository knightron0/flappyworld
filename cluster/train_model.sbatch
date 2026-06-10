#!/bin/bash
#SBATCH --job-name=flappy-flatlm-v2-100k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs /scratch/gilbreth/mangla/flappybird/checkpoints
source /scratch/gilbreth/mangla/venvs/flappyworld/bin/activate

DATA=${DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_visible_pipes_v2.jsonl}
OUT=${OUT:-/scratch/gilbreth/mangla/flappybird/checkpoints/tokenized_flat_lm_visible_pipes_v2_death_w50_100k}
STEPS=${STEPS:-100000}
DEATH_WEIGHT=${DEATH_WEIGHT:-50}
DONE_POSITIVE_WEIGHT=${DONE_POSITIVE_WEIGHT:-50}

python -m scripts.train_model \
  --data "$DATA" \
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
  --death-weight "$DEATH_WEIGHT" \
  --done-positive-weight "$DONE_POSITIVE_WEIGHT"

echo "data=$DATA"
echo "checkpoint_dir=$OUT"
