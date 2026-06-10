"""Convert raw Flappy LM JSONL into flat tokens with explicit pipe visibility.

The raw environment encodes hidden future pipes as:

    x = screen_width, top = 0, bottom = screen_height

and keeps off-screen-left pipes at x <= 0 until they respawn on the right.
Those left-edge frames were previously tokenized as pipe_x_000 with
present_1, which leaves a fake pipe stuck on the left edge. This script
marks both cases as hidden:

    pipe1_present_0 pipe1_x_hidden pipe1_gap_hidden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512


def quantize_unit(value: float, bins: int) -> int:
    value = max(0.0, min(1.0, float(value)))
    return int(round(value * (bins - 1)))


def pipe_present(obs: list[float], pipe_idx: int, eps: float) -> bool:
    offset = pipe_idx * 3
    x = float(obs[offset])
    top = float(obs[offset + 1])
    bottom = float(obs[offset + 2])
    if x >= 1.0 - eps and top <= eps and bottom >= 1.0 - eps:
        return False
    if x <= eps:
        return False
    return True


def pipe_gap_bin(obs: list[float], pipe_idx: int, bins: int) -> int:
    offset = pipe_idx * 3
    gap_center = 0.5 * (float(obs[offset + 1]) + float(obs[offset + 2]))
    return quantize_unit(gap_center, bins)


def frame_values(
    obs: list[float],
    action: int,
    prev_frame: dict | None,
    done: bool,
    bird_y_bins: int,
    pipe_x_bins: int,
    pipe_gap_bins: int,
    respawn_threshold_bins: int,
    hidden_eps: float,
) -> dict:
    frame = {
        "bird_y": quantize_unit(float(obs[9]), bird_y_bins),
        "pipes": [],
        "respawn": 0,
        "done": int(done),
        "action": int(action),
    }
    for pipe_idx in range(2):
        present = pipe_present(obs, pipe_idx, hidden_eps)
        pipe = {
            "present": int(present),
            "x": quantize_unit(float(obs[pipe_idx * 3]), pipe_x_bins) if present else None,
            "gap": pipe_gap_bin(obs, pipe_idx, pipe_gap_bins) if present else None,
        }
        if present and prev_frame is not None:
            prev_pipe = prev_frame["pipes"][pipe_idx]
            if not prev_pipe["present"] or pipe["x"] - prev_pipe["x"] >= respawn_threshold_bins:
                frame["respawn"] = 1
        frame["pipes"].append(pipe)
    return frame


def frame_tokens(frame: dict) -> list[str]:
    tokens = ["<frame>", f"bird_y_{frame['bird_y']:03d}"]
    for pipe_idx, pipe in enumerate(frame["pipes"]):
        tokens.append(f"pipe{pipe_idx}_present_{pipe['present']}")
        if pipe["present"]:
            tokens.append(f"pipe{pipe_idx}_x_{pipe['x']:03d}")
            tokens.append(f"pipe{pipe_idx}_gap_{pipe['gap']:03d}")
        else:
            tokens.append(f"pipe{pipe_idx}_x_hidden")
            tokens.append(f"pipe{pipe_idx}_gap_hidden")
    tokens.extend(
        [
            f"respawn_{frame['respawn']}",
            f"done_{frame['done']}",
            f"action_{frame['action']}",
            "</frame>",
        ]
    )
    return tokens


def convert_record(record: dict, args: argparse.Namespace) -> dict | None:
    observations = record["observations"]
    actions = record["actions"]
    dones = record.get("dones", [False] * len(actions))
    length = min(len(observations), len(actions), len(dones))
    if length < 1:
        return None

    tokens = ["<bos>"]
    frames = []
    prev_frame = None
    for idx in range(length):
        done_from_prev = bool(dones[idx - 1]) if idx > 0 else False
        frame = frame_values(
            observations[idx],
            int(actions[idx]),
            prev_frame,
            done_from_prev,
            args.bird_y_bins,
            args.pipe_x_bins,
            args.pipe_gap_bins,
            args.respawn_threshold_bins,
            args.hidden_eps,
        )
        tokens.extend(frame_tokens(frame))
        frames.append(frame)
        prev_frame = frame

    rewards = record.get("rewards", [0.0] * length)
    terminal_death = bool(dones[length - 1]) and float(rewards[length - 1]) <= -1.0
    if terminal_death:
        terminal_observations = record.get("terminal_observations")
        if not terminal_observations:
            terminal_obs = record.get("terminal_observation")
            terminal_observations = [terminal_obs] if terminal_obs is not None else []
        if not terminal_observations:
            terminal_observations = [observations[length - 1]]
        for hold_idx, terminal_obs in enumerate(terminal_observations):
            death_frame = frame_values(
                terminal_obs,
                int(actions[length - 1]),
                prev_frame,
                True,
                args.bird_y_bins,
                args.pipe_x_bins,
                args.pipe_gap_bins,
                args.respawn_threshold_bins,
                args.hidden_eps,
            )
            death_frame["respawn"] = 0
            death_frame["done"] = 1
            tokens.extend(frame_tokens(death_frame))
            frames.append(death_frame)
            prev_frame = death_frame
        tokens.append("<DEATH>")

    tokens.append("<eos>")
    output = dict(record)
    output["format"] = "visible_pipe_flat_lm_v1"
    output["tokens"] = tokens
    output["visible_pipe_frames"] = frames
    output["visible_pipe_tokenizer"] = {
        "bird_y_bins": args.bird_y_bins,
        "pipe_x_bins": args.pipe_x_bins,
        "pipe_gap_bins": args.pipe_gap_bins,
        "respawn_threshold_bins": args.respawn_threshold_bins,
        "hidden_eps": args.hidden_eps,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--bird-y-bins", type=int, default=128)
    parser.add_argument("--pipe-x-bins", type=int, default=128)
    parser.add_argument("--pipe-gap-bins", type=int, default=96)
    parser.add_argument("--respawn-threshold-bins", type=int, default=8)
    parser.add_argument("--hidden-eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    hidden_counts = [0, 0]
    total_frames = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if args.max_records is not None and line_idx >= args.max_records:
                break
            converted = convert_record(json.loads(line), args)
            if converted is None:
                continue
            for frame in converted["visible_pipe_frames"]:
                total_frames += 1
                for pipe_idx, pipe in enumerate(frame["pipes"]):
                    hidden_counts[pipe_idx] += int(not pipe["present"])
            dst.write(json.dumps(converted) + "\n")
            written += 1
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"records={written} frames={total_frames}")
    print(f"hidden_pipe0={hidden_counts[0]} hidden_pipe1={hidden_counts[1]}")


if __name__ == "__main__":
    main()
