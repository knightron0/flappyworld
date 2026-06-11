If you’ve read [this blog post](https://www.sarthakmangla.com/blog/wccl), you already know my favorite hackathon archetype is to build overengineered solutions to problems that don't exist. This started back in July 2025 at the KP Hackathon with Shobhit Agarwal. We'd seen demos of 


--- 

--- 

## One-Line Summary
We turn Flappy Bird gameplay into structured token sequences, train a flat autoregressive world model on them, and serve rollouts back in the browser.

## Goals
- Explain the problem clearly before getting into implementation details.
- Show the tokenized frame format we actually use.
- Show the data collection mix we actually used.
- Summarize the training setup and the main losses.
- Show how the browser player loads and runs the checkpoint.

## Suggested Structure

### 1. Problem Setup
- Predict the next frame from a compact token stream.
- No game engine or hand-written physics in the rollout.
- Goal: playable behavior in the browser.

### 2. Data Representation
- `bird_y`, `pipe0_x`, `pipe0_gap`, `pipe1_x`, `pipe1_gap`, `respawn`, `done`, `action`
- Hidden pipes become `pipe*_present_0`, `pipe*_x_hidden`, `pipe*_gap_hidden`
- Terminal deaths get extra hold frames before `<DEATH>`
- Raw records also keep observations, actions, rewards, dones, and scores

### 3. Data Collection
- First pass comes before training.
- Episode policies include `ppo`, `random`, `ppo_noisy`, `idle`, `random_then_idle`, and `ppo_then_idle`.
- `--terminal-hold-frames` records the frozen terminal state after death.
- Output is JSONL, one episode per record.

### 4. Training Recipes and Models
- Flat token LM over compact frames.
- Loss terms are weighted around numeric bins, respawn, done, and action.
- `global_rope` is used for browser serving with KV cache.
- Rollout loss is available in the training loop.

### 5. First Results and Failures
- Short rollouts are plausible.
- Rare `done` events are easy to miss.
- Respawn jumps are discontinuous.
- Hidden/offscreen pipes need explicit handling.
- Long rollouts drift.

### 6. Data Got Better
- Add failure-heavy episodes after the first failures show up.
- Keep raw JSONL so preprocessing can change later.
- Terminal hold frames make `done` easier to learn.

### 7. Final Training Result
- Best setup is the one that matches the final tokenization and loss weighting.
- Mention the main ablations briefly.
- Keep the result tied to rollout quality, not just validation loss.

### 8. Inference and Serving
- Browser player loads `manifest.json` plus ONNX prefill/decode models.
- Seeding comes from real data lines and state indices.
- The player keeps a KV cache while stepping autoregressively.
- The UI exposes the token stream, `respawn`, and `done`.

### 9. Lessons Learned
- Explicit token structure matters.
- Hidden pipes need explicit visibility handling.
- Terminal frames matter.
- Data quality mattered more than model size.

## Concrete Subsections to Fill In

### Data Representation
- Current token order per frame.
- Why `bird_y`, `pipe_x`, and `gap` are discretized.
- How `done`, `respawn`, and `action` are encoded.
- Why `bird_rot` and `score` are useful additions.

### Collection
- Episode policy mix.
- Why intentionally bad episodes matter.
- How we preserve terminal collision frames.
- Why raw JSONL is kept before preprocessing.

### Training
- Loss weights.
- Numeric vs categorical heads.
- History length / block size.
- Best sub-500k configuration.
- Whether rollout loss helped.

### Serving
- How the model is prompted from real data.
- How the browser render loop consumes checkpoints.
- What an interactive rollout looks like.

## Draft Notes
- Keep the tone practical.
- Avoid overexplaining basic Flappy Bird mechanics.
- Focus on representation choices and what they buy you.
- Include one or two concrete token examples.
- Include one figure or table if possible:
  - raw observation -> tokenized frame
  - model comparison table

## Open Questions
- Should the blog emphasize the flat LM path or the browser player path?
- Should score and rotation be presented as the final representation or as an experiment?
- Do we want to show one dataset example before and after preprocessing?
