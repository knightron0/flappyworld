#!/bin/bash
#SBATCH --job-name=flappy-flatlm-preprocess-terminal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=32G

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs /scratch/gilbreth/mangla/flappybird/datasets
source /scratch/gilbreth/mangla/venvs/flappyworld/bin/activate

RAW_DATA=${RAW_DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal.jsonl}
VISIBLE_DATA=${VISIBLE_DATA:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal_visible_pipes_v3_death_after_collision.jsonl}

if [[ ! -s "$RAW_DATA" ]]; then
  echo "missing raw dataset: $RAW_DATA" >&2
  exit 1
fi

python -m scripts.preprocess_data \
  --input "$RAW_DATA" \
  --output "$VISIBLE_DATA"

echo "raw_data=$RAW_DATA"
echo "visible_data=$VISIBLE_DATA"
