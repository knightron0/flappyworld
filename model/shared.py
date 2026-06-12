"""Shared frame/token utilities for the flat autoregressive Flappy LM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameTokens:
    bird_y: int
    pipe0_x: int
    pipe0_gap: int
    pipe1_x: int
    pipe1_gap: int
    respawn: int
    done: int
    action: int


@dataclass
class TokenizerConfig:
    bird_y_bins: int
    pipe_x_bins: int
    pipe_gap_bins: int
    respawn_threshold_bins: int


class Episode:
    def __init__(self, frames: list[FrameTokens], final_done: bool):
        self.frames = frames
        self.final_done = final_done


def quantize_unit(value: float, bins: int) -> int:
    value = max(0.0, min(1.0, float(value)))
    return int(round(value * (bins - 1)))


def pipe_gap_center(obs: list[float], pipe_idx: int) -> float:
    offset = pipe_idx * 3
    return 0.5 * (float(obs[offset + 1]) + float(obs[offset + 2]))


def obs_to_frame(obs: list[float], action: int, cfg: TokenizerConfig) -> FrameTokens:
    if len(obs) <= 9:
        raise ValueError(f"Expected at least 10 observation fields, got {len(obs)}")
    return FrameTokens(
        bird_y=quantize_unit(float(obs[9]), cfg.bird_y_bins),
        pipe0_x=quantize_unit(float(obs[0]), cfg.pipe_x_bins),
        pipe0_gap=quantize_unit(pipe_gap_center(obs, 0), cfg.pipe_gap_bins),
        pipe1_x=quantize_unit(float(obs[3]), cfg.pipe_x_bins),
        pipe1_gap=quantize_unit(pipe_gap_center(obs, 1), cfg.pipe_gap_bins),
        respawn=0,
        done=0,
        action=int(action),
    )


def mark_transition_events(prev: FrameTokens, cur: FrameTokens, done_from_prev: bool, cfg: TokenizerConfig) -> FrameTokens:
    pipe0_jump = cur.pipe0_x - prev.pipe0_x
    pipe1_jump = cur.pipe1_x - prev.pipe1_x
    respawn = int(pipe0_jump >= cfg.respawn_threshold_bins or pipe1_jump >= cfg.respawn_threshold_bins)
    return FrameTokens(
        bird_y=cur.bird_y,
        pipe0_x=cur.pipe0_x,
        pipe0_gap=cur.pipe0_gap,
        pipe1_x=cur.pipe1_x,
        pipe1_gap=cur.pipe1_gap,
        respawn=respawn,
        done=int(done_from_prev),
        action=cur.action,
    )


def load_episodes(path: Path, tokenizer_cfg: TokenizerConfig, max_episodes: int | None = None) -> list[Episode]:
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_episodes is not None and line_idx >= max_episodes:
                break
            record = json.loads(line)
            observations = record["observations"]
            actions = record["actions"]
            dones = record.get("dones", [False] * len(actions))
            length = min(len(observations), len(actions), len(dones))
            if length == 0:
                continue
            frames: list[FrameTokens] = []
            partial = [obs_to_frame(observations[idx], actions[idx], tokenizer_cfg) for idx in range(length)]
            frames.append(partial[0])
            for idx in range(1, length):
                frames.append(mark_transition_events(partial[idx - 1], partial[idx], bool(dones[idx - 1]), tokenizer_cfg))
            final_done = bool(dones[length - 1])
            episodes.append(Episode(frames, final_done))
    return episodes
