#!/bin/bash
#SBATCH --job-name=flappy-flatlm-collect-terminal
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
mkdir -p logs /scratch/gilbreth/mangla/flappybird/datasets
source /scratch/gilbreth/mangla/venvs/flappyworld/bin/activate

TARGET_EPISODES=${TARGET_EPISODES:-50000}
SEED=${SEED:-2026}
OUTPUT=${OUTPUT:-/scratch/gilbreth/mangla/flappybird/datasets/flappy_flatlm_mixed_bad_idle_50k_terminal.jsonl}
MODEL=${MODEL:-flappy_bird_ppo}
POLICY_MIX=${POLICY_MIX:-ppo:0.25,random:0.2,ppo_noisy:0.15,random_then_idle:0.2,ppo_then_idle:0.15,idle:0.05}
IDLE_AFTER_STEPS=${IDLE_AFTER_STEPS:-12}
TERMINAL_HOLD_FRAMES=${TERMINAL_HOLD_FRAMES:-1}

EXISTING=0
if [[ -s "$OUTPUT" ]]; then
  EXISTING=$(wc -l < "$OUTPUT")
fi
REMAINING=$((TARGET_EPISODES - EXISTING))
if (( REMAINING <= 0 )); then
  echo "already_have_episodes=$EXISTING target=$TARGET_EPISODES"
  exit 0
fi

COLLECT_ARGS=(--num-games "$REMAINING" --start-episode "$EXISTING")
if (( EXISTING > 0 )); then
  COLLECT_ARGS+=(--append)
  SEED=$((SEED + EXISTING))
fi

python -m scripts.collect_data \
  "${COLLECT_ARGS[@]}" \
  --episode-policy-mix "$POLICY_MIX" \
  --ppo-noisy-epsilon 0.15 \
  --idle-after-steps "$IDLE_AFTER_STEPS" \
  --max-episode-steps 5000 \
  --terminal-hold-frames "$TERMINAL_HOLD_FRAMES" \
  --model-path "$MODEL" \
  --seed "$SEED" \
  --output "$OUTPUT"

FINAL_COUNT=$(wc -l < "$OUTPUT")
echo "dataset=$OUTPUT"
echo "existing=$EXISTING added=$REMAINING final=$FINAL_COUNT"
echo "terminal_hold_frames=$TERMINAL_HOLD_FRAMES"
