"""Collect tokenized Flappy Bird trajectories for language-model world models.

Each episode is saved as one JSONL record containing:
  - raw normalized observations/actions/rewards for numeric experiments
  - a flat token sequence suitable for decoder-only next-token training

The token sequence is organized as:
  <bos> state(obs_0) action_0 reward/done state(obs_1) action_1 ...
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium
import numpy as np
from stable_baselines3 import PPO

import flappy_bird_gymnasium  # noqa: F401 - registers the environment


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
NUM_PIPES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--ppo-ratio", type=float, default=0.5)
    parser.add_argument(
        "--episode-policy-mix",
        type=str,
        default=None,
        help=(
            "Episode-level policy mix, e.g. ppo:0.7,random:0.2,ppo_noisy:0.1,idle:0.1. "
            "Also supports random_then_idle and ppo_then_idle for intentionally bad episodes."
        ),
    )
    parser.add_argument("--idle-after-steps", type=int, default=12)
    parser.add_argument("--ppo-noisy-epsilon", type=float, default=0.15)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="flappy_bird_ppo")
    parser.add_argument("--output", type=str, default="dataset/flappy_lm_trajectories.jsonl")
    parser.add_argument(
        "--terminal-hold-frames",
        type=int,
        default=1,
        help=(
            "On death, also record post-collision observation(s) from the env. "
            "The first frame is the actual collision geometry; extra holds repeat the frozen terminal state."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new episodes to an existing JSONL file instead of overwriting it.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index written into each record's episode field (use when resuming collection).",
    )
    return parser.parse_args()


def quantize_screen(value: float, scale: int) -> int:
    """Convert a normalized screen coordinate back to an integer screen bucket."""
    return int(round(float(np.clip(value, 0.0, 1.0)) * scale))


def quantize_rotation(value: float) -> int:
    return int(round(float(value) * 90))


def observation_tokens(obs: np.ndarray) -> list[str]:
    tokens = []
    for pipe_idx in range(NUM_PIPES):
        offset = pipe_idx * 3
        pipe_x = quantize_screen(obs[offset], SCREEN_WIDTH)
        gap_top = quantize_screen(obs[offset + 1], SCREEN_HEIGHT)
        gap_bottom = quantize_screen(obs[offset + 2], SCREEN_HEIGHT)
        tokens.extend(
            [
                f"p{pipe_idx}_x_{pipe_x:03d}",
                f"p{pipe_idx}_top_{gap_top:03d}",
                f"p{pipe_idx}_bottom_{gap_bottom:03d}",
            ]
        )

    tokens.extend(
        [
            f"bird_y_{quantize_screen(obs[9], SCREEN_HEIGHT):03d}",
            f"bird_rot_{quantize_rotation(obs[11]):+04d}",
        ]
    )
    return tokens


def reward_token(reward: float) -> str:
    if reward >= 1.0:
        return "R_PIPE"
    if reward <= -1.0:
        return "R_DEAD"
    if reward < 0.0:
        return "R_BAD"
    return "R_ALIVE"


def parse_episode_policy_mix(value: str | None) -> list[tuple[str, float]] | None:
    if value is None:
        return None
    items = []
    total = 0.0
    for chunk in value.split(","):
        name, weight_text = chunk.split(":", 1)
        name = name.strip()
        weight = float(weight_text)
        if name not in {"ppo", "random", "ppo_noisy", "idle", "random_then_idle", "ppo_then_idle"}:
            raise ValueError(f"Unknown episode policy {name!r}")
        if weight < 0:
            raise ValueError("Episode policy weights must be non-negative")
        items.append((name, weight))
        total += weight
    if total <= 0:
        raise ValueError("Episode policy mix must have positive total weight")
    return [(name, weight / total) for name, weight in items]


def choose_episode_policy(policy_mix: list[tuple[str, float]], rng: np.random.Generator) -> str:
    names = [name for name, _weight in policy_mix]
    probs = [weight for _name, weight in policy_mix]
    return str(rng.choice(names, p=probs))


def collect_episode(
    env,
    model: PPO | None,
    ppo_ratio: float,
    rng: np.random.Generator,
    episode_policy: str | None = None,
    ppo_noisy_epsilon: float = 0.15,
    max_episode_steps: int | None = None,
    idle_after_steps: int = 12,
    terminal_hold_frames: int = 1,
):
    obs, _ = env.reset()
    observations = []
    actions = []
    action_sources = []
    rewards = []
    dones = []
    scores = []
    terminal_observation = None
    terminal_observations: list[list[float]] = []

    tokens = ["<bos>", *observation_tokens(obs)]

    while True:
        if max_episode_steps is not None and len(actions) >= max_episode_steps:
            break
        force_idle = episode_policy == "idle" or episode_policy in {"random_then_idle", "ppo_then_idle"} and len(actions) >= idle_after_steps
        if force_idle:
            action = 0
            source = "idle"
        elif episode_policy == "ppo":
            if model is None:
                raise ValueError("episode_policy='ppo' requires a loaded PPO model")
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            source = "ppo"
        elif episode_policy == "random":
            action = int(env.action_space.sample())
            source = "random"
        elif episode_policy == "ppo_noisy":
            if model is None:
                raise ValueError("episode_policy='ppo_noisy' requires a loaded PPO model")
            if rng.random() < ppo_noisy_epsilon:
                action = int(env.action_space.sample())
                source = "random"
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                source = "ppo"
        elif episode_policy == "random_then_idle":
            action = int(env.action_space.sample())
            source = "random_pre_idle"
        elif episode_policy == "ppo_then_idle":
            if model is None:
                raise ValueError("episode_policy='ppo_then_idle' requires a loaded PPO model")
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            source = "ppo_pre_idle"
        elif model is not None and rng.random() < ppo_ratio:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            source = "ppo"
        else:
            action = int(env.action_space.sample())
            source = "random"

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        observations.append(obs.astype(np.float32).tolist())
        actions.append(action)
        action_sources.append(source)
        rewards.append(float(reward))
        dones.append(done)
        scores.append(int(info.get("score", 0)))

        tokens.append("A_FLAP" if action == 1 else "A_IDLE")
        tokens.append(reward_token(float(reward)))
        if done:
            if float(reward) <= -1.0:
                hold = max(1, int(terminal_hold_frames))
                terminal_obs = next_obs.astype(np.float32)
                terminal_observation = terminal_obs.tolist()
                terminal_observations = [terminal_observation]
                tokens.extend(observation_tokens(terminal_obs))
                # After terminal, the env freezes; extra holds repeat collision geometry.
                frozen = terminal_obs
                for _ in range(hold - 1):
                    frozen_obs, frozen_reward, frozen_term, _, _ = env.step(0)
                    if not frozen_term:
                        break
                    terminal_observations.append(frozen_obs.astype(np.float32).tolist())
                    tokens.extend(observation_tokens(frozen_obs))
            tokens.append("<done>")
            break

        tokens.extend(observation_tokens(next_obs))
        obs = next_obs

    episode = {
        "tokens": tokens,
        "observations": observations,
        "actions": actions,
        "action_sources": action_sources,
        "rewards": rewards,
        "dones": dones,
        "scores": scores,
        "final_score": scores[-1] if scores else 0,
        "steps": len(actions),
        "episode_policy": episode_policy or f"per_step_ppo_ratio_{ppo_ratio:g}",
        "episode_policy_config": {
            "ppo_ratio": ppo_ratio,
            "ppo_noisy_epsilon": ppo_noisy_epsilon,
            "terminal_hold_frames": terminal_hold_frames,
        },
    }
    if terminal_observation is not None:
        episode["terminal_observation"] = terminal_observation
        episode["terminal_observations"] = terminal_observations
    return episode


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ppo_ratio <= 1.0:
        raise ValueError("--ppo-ratio must be between 0 and 1")
    if not 0.0 <= args.ppo_noisy_epsilon <= 1.0:
        raise ValueError("--ppo-noisy-epsilon must be between 0 and 1")
    episode_policy_mix = parse_episode_policy_mix(args.episode_policy_mix)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    needs_ppo = args.ppo_ratio > 0 or episode_policy_mix is not None and any(name != "random" for name, _ in episode_policy_mix)
    if needs_ppo:
        model_file = Path(args.model_path)
        if not model_file.with_suffix(".zip").exists() and not model_file.exists():
            raise FileNotFoundError(
                f"PPO model not found at {args.model_path}. "
                "Use --ppo-ratio 0 for random-only collection."
            )
        model = PPO.load(args.model_path)

    rng = np.random.default_rng(args.seed)
    env = gymnasium.make("FlappyBird-v0", use_lidar=False)

    total_steps = 0
    scores = []
    policy_counts: dict[str, int] = {}
    file_mode = "a" if args.append else "w"
    with output_path.open(file_mode, encoding="utf-8") as f:
        for game_idx in range(args.num_games):
            episode_policy = choose_episode_policy(episode_policy_mix, rng) if episode_policy_mix is not None else None
            episode = collect_episode(
                env,
                model,
                args.ppo_ratio,
                rng,
                episode_policy,
                args.ppo_noisy_epsilon,
                args.max_episode_steps,
                args.idle_after_steps,
                args.terminal_hold_frames,
            )
            episode["episode"] = args.start_episode + game_idx
            f.write(json.dumps(episode) + "\n")

            total_steps += episode["steps"]
            scores.append(episode["final_score"])
            policy_counts[episode["episode_policy"]] = policy_counts.get(episode["episode_policy"], 0) + 1
            print(
                f"game={game_idx + 1}/{args.num_games} "
                f"episode={episode['episode']} "
                f"policy={episode['episode_policy']} "
                f"score={episode['final_score']} steps={episode['steps']}"
            )

    env.close()
    print(f"saved={output_path} mode={file_mode}")
    print(f"new_episodes={args.num_games} start_episode={args.start_episode} total_steps={total_steps}")
    print(f"avg_score={np.mean(scores):.2f} max_score={np.max(scores)}")
    print(f"policy_counts={policy_counts}")


if __name__ == "__main__":
    main()
