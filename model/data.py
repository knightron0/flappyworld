"""Train a typed token autoregressive Flappy world model.

This is intentionally separate from the flat LM experiments in the repository.
Each frame is represented as typed tokens:

    bird_y_bin, pipe0_x_bin, pipe0_gap_bin, pipe1_x_bin, pipe1_gap_bin,
    respawn, done, action

The model conditions on a short history of previous frames and actions. Bird
velocity is not part of the state; it has to be inferred from recent bird_y
tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCREEN_WIDTH = 288.0
SCREEN_HEIGHT = 512.0
PIPE_FIELDS = ("pipe0", "pipe1")


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


@dataclass
class ModelConfig:
    history_size: int
    bird_y_bins: int
    pipe_x_bins: int
    pipe_gap_bins: int
    respawn_threshold_bins: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float


@dataclass
class LossWeights:
    bird_y: float
    pipe_x: float
    gap_respawn: float
    respawn: float
    done: float
    action: float


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


def obs_to_partial_frame(obs: list[float], action: int, cfg: TokenizerConfig) -> FrameTokens:
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


def pipe_respawn_masks(prev: torch.Tensor, target: torch.Tensor, cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    pipe0_jump = target[:, 1] - prev[:, 1]
    pipe1_jump = target[:, 3] - prev[:, 3]
    return pipe0_jump >= cfg.respawn_threshold_bins, pipe1_jump >= cfg.respawn_threshold_bins


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
            if length < 2:
                continue

            base_frames = [
                obs_to_partial_frame(observations[idx], int(actions[idx]), tokenizer_cfg)
                for idx in range(length)
            ]
            frames = [base_frames[0]]
            for idx in range(1, length):
                frames.append(mark_transition_events(base_frames[idx - 1], base_frames[idx], bool(dones[idx - 1]), tokenizer_cfg))
            episodes.append(Episode(frames, bool(dones[length - 1])))

    if not episodes:
        raise ValueError(f"No usable episodes found in {path}")
    return episodes


def split_episodes(episodes: list[Episode], train_frac: float = 0.9) -> tuple[list[Episode], list[Episode]]:
    split = max(1, int(len(episodes) * train_frac))
    if split >= len(episodes):
        return episodes, episodes
    return episodes[:split], episodes[split:]


def frame_to_tensor(frame: FrameTokens) -> torch.Tensor:
    return torch.tensor(
        [
            frame.bird_y,
            frame.pipe0_x,
            frame.pipe0_gap,
            frame.pipe1_x,
            frame.pipe1_gap,
            frame.respawn,
            frame.done,
            frame.action,
        ],
        dtype=torch.long,
    )


class FrameTransitionDataset(Dataset):
    def __init__(self, episodes: list[Episode], history_size: int):
        self.episodes = episodes
        self.history_size = history_size
        self.items: list[tuple[int, int, bool]] = []
        for episode_idx, episode in enumerate(episodes):
            for step_idx in range(len(episode.frames) - 1):
                self.items.append((episode_idx, step_idx, False))
            if episode.final_done:
                self.items.append((episode_idx, len(episode.frames) - 1, True))
        if not self.items:
            raise ValueError("No frame transitions available")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        episode_idx, step_idx, is_terminal = self.items[idx]
        frames = self.episodes[episode_idx].frames
        start = max(0, step_idx - self.history_size + 1)
        context = [frame_to_tensor(frame) for frame in frames[start : step_idx + 1]]
        pad = self.history_size - len(context)
        if pad:
            context = [frame_to_tensor(frames[0])] * pad + context
        if is_terminal:
            current = frames[step_idx]
            target = FrameTokens(
                bird_y=current.bird_y,
                pipe0_x=current.pipe0_x,
                pipe0_gap=current.pipe0_gap,
                pipe1_x=current.pipe1_x,
                pipe1_gap=current.pipe1_gap,
                respawn=0,
                done=1,
                action=current.action,
            )
            return torch.stack(context, dim=0), frame_to_tensor(target), torch.tensor(0.0, dtype=torch.float32)
        return torch.stack(context, dim=0), frame_to_tensor(frames[step_idx + 1]), torch.tensor(1.0, dtype=torch.float32)


class FrameRolloutDataset(Dataset):
    def __init__(self, episodes: list[Episode], history_size: int, rollout_steps: int):
        self.episodes = episodes
        self.history_size = history_size
        self.rollout_steps = rollout_steps
        self.items: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(episodes):
            for step_idx in range(max(0, len(episode.frames) - rollout_steps)):
                self.items.append((episode_idx, step_idx))
        if not self.items:
            raise ValueError(f"No {rollout_steps}-step rollout windows available")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        episode_idx, step_idx = self.items[idx]
        frames = self.episodes[episode_idx].frames
        start = max(0, step_idx - self.history_size + 1)
        context = [frame_to_tensor(frame) for frame in frames[start : step_idx + 1]]
        pad = self.history_size - len(context)
        if pad:
            context = [frame_to_tensor(frames[0])] * pad + context
        targets = [frame_to_tensor(frame) for frame in frames[step_idx + 1 : step_idx + 1 + self.rollout_steps]]
        return torch.stack(context, dim=0), torch.stack(targets, dim=0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(batch, time, channels))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TokenizedARWorldModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bird_y_emb = nn.Embedding(config.bird_y_bins, config.n_embd)
        self.pipe_x_emb = nn.Embedding(config.pipe_x_bins, config.n_embd)
        self.pipe_gap_emb = nn.Embedding(config.pipe_gap_bins, config.n_embd)
        self.binary_emb = nn.Embedding(2, config.n_embd)
        self.action_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Embedding(config.history_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.bird_y_head = nn.Linear(config.n_embd, config.bird_y_bins)
        self.pipe0_x_head = nn.Linear(config.n_embd, config.pipe_x_bins)
        self.pipe1_x_head = nn.Linear(config.n_embd, config.pipe_x_bins)
        self.pipe0_gap_head = nn.Linear(config.n_embd, config.pipe_gap_bins)
        self.pipe1_gap_head = nn.Linear(config.n_embd, config.pipe_gap_bins)
        self.respawn_head = nn.Linear(config.n_embd, 1)
        self.done_head = nn.Linear(config.n_embd, 1)
        self.action_head = nn.Linear(config.n_embd, 2)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        _, time, _ = frames.shape
        pos = torch.arange(time, device=frames.device)
        x = (
            self.bird_y_emb(frames[:, :, 0])
            + self.pipe_x_emb(frames[:, :, 1])
            + self.pipe_gap_emb(frames[:, :, 2])
            + self.pipe_x_emb(frames[:, :, 3])
            + self.pipe_gap_emb(frames[:, :, 4])
            + self.binary_emb(frames[:, :, 5])
            + self.binary_emb(frames[:, :, 6])
            + self.action_emb(frames[:, :, 7])
            + self.pos_emb(pos)
        )
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x[:, -1])
        return {
            "bird_y": self.bird_y_head(h),
            "pipe0_x": self.pipe0_x_head(h),
            "pipe1_x": self.pipe1_x_head(h),
            "pipe0_gap": self.pipe0_gap_head(h),
            "pipe1_gap": self.pipe1_gap_head(h),
            "respawn": self.respawn_head(h).squeeze(-1),
            "done": self.done_head(h).squeeze(-1),
            "action": self.action_head(h),
        }


def geometry_aware_ce(logits: torch.Tensor, targets: torch.Tensor, sigma: float, window: int) -> torch.Tensor:
    if sigma <= 0.0:
        return F.cross_entropy(logits, targets, reduction="none")
    bins = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
    distances = (bins.unsqueeze(0) - targets.to(logits.dtype).unsqueeze(1)).abs()
    if window > 0:
        mask = distances <= float(window)
    else:
        mask = torch.ones_like(distances, dtype=torch.bool)
    probs = torch.exp(-0.5 * (distances / sigma).pow(2)).masked_fill(~mask, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def loss_and_metrics(
    model: TokenizedARWorldModel,
    batch,
    device: torch.device,
    weights: LossWeights,
    numeric_sigma: float,
    numeric_window: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    context, target, valid_state = [item.to(device) for item in batch]
    valid_mask = valid_state > 0.5
    logits = model(context)
    prev = context[:, -1]
    pipe0_respawn, pipe1_respawn = pipe_respawn_masks(prev, target, model.config)

    bird_y_loss = masked_mean(geometry_aware_ce(logits["bird_y"], target[:, 0], numeric_sigma, numeric_window), valid_mask)
    pipe0_x_loss = masked_mean(geometry_aware_ce(logits["pipe0_x"], target[:, 1], numeric_sigma, numeric_window), valid_mask)
    pipe1_x_loss = masked_mean(geometry_aware_ce(logits["pipe1_x"], target[:, 3], numeric_sigma, numeric_window), valid_mask)
    pipe0_gap_loss = masked_mean(F.cross_entropy(logits["pipe0_gap"], target[:, 2], reduction="none"), pipe0_respawn & valid_mask)
    pipe1_gap_loss = masked_mean(F.cross_entropy(logits["pipe1_gap"], target[:, 4], reduction="none"), pipe1_respawn & valid_mask)
    respawn_loss = masked_mean(
        F.binary_cross_entropy_with_logits(logits["respawn"], target[:, 5].float(), reduction="none"),
        valid_mask,
    )
    done_loss = F.binary_cross_entropy_with_logits(logits["done"], target[:, 6].float())
    action_loss = masked_mean(F.cross_entropy(logits["action"], target[:, 7], reduction="none"), valid_mask)

    loss = (
        weights.bird_y * bird_y_loss
        + weights.pipe_x * (pipe0_x_loss + pipe1_x_loss)
        + weights.gap_respawn * (pipe0_gap_loss + pipe1_gap_loss)
        + weights.respawn * respawn_loss
        + weights.done * done_loss
        + weights.action * action_loss
    )

    with torch.no_grad():
        metrics = {
            "loss": float(loss.item()),
            "bird_y_loss": float(bird_y_loss.item()),
            "pipe0_x_loss": float(pipe0_x_loss.item()),
            "pipe1_x_loss": float(pipe1_x_loss.item()),
            "pipe0_gap_loss": float(pipe0_gap_loss.item()),
            "pipe1_gap_loss": float(pipe1_gap_loss.item()),
            "respawn_loss": float(respawn_loss.item()),
            "done_loss": float(done_loss.item()),
            "action_loss": float(action_loss.item()),
            "bird_y_bin_mae": float(masked_mean((logits["bird_y"].argmax(dim=-1) - target[:, 0]).abs().float(), valid_mask).item()),
            "pipe_x_bin_mae": float(
                0.5
                * (
                    masked_mean((logits["pipe0_x"].argmax(dim=-1) - target[:, 1]).abs().float(), valid_mask)
                    + masked_mean((logits["pipe1_x"].argmax(dim=-1) - target[:, 3]).abs().float(), valid_mask)
                ).item()
            ),
            "respawn_acc": float(masked_mean(((logits["respawn"] > 0).long() == target[:, 5]).float(), valid_mask).item()),
            "done_acc": float(((logits["done"] > 0).long() == target[:, 6]).float().mean().item()),
            "action_acc": float(masked_mean((logits["action"].argmax(dim=-1) == target[:, 7]).float(), valid_mask).item()),
            "pipe0_respawn_count": float(pipe0_respawn.float().sum().item()),
            "pipe1_respawn_count": float(pipe1_respawn.float().sum().item()),
        }
    return loss, metrics


@torch.no_grad()
def estimate_metrics(
    model: TokenizedARWorldModel,
    loader: DataLoader,
    device: torch.device,
    eval_iters: int,
    weights: LossWeights,
    numeric_sigma: float,
    numeric_window: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    iterator = iter(loader)
    for _ in range(eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        _loss, metrics = loss_and_metrics(model, batch, device, weights, numeric_sigma, numeric_window)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    model.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def parse_rollout_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("--rollout-eval-horizons must contain positive integers")
    return horizons


def next_frame_prediction(
    logits: dict[str, torch.Tensor],
    prev: torch.Tensor,
    target_action: torch.Tensor | None,
    respawn_threshold_bins: int,
) -> torch.Tensor:
    bird_y = logits["bird_y"].argmax(dim=-1)
    pipe0_x = logits["pipe0_x"].argmax(dim=-1)
    pipe1_x = logits["pipe1_x"].argmax(dim=-1)
    predicted_respawn = (logits["respawn"] > 0).long()
    pipe0_respawn = (pipe0_x - prev[:, 1]) >= respawn_threshold_bins
    pipe1_respawn = (pipe1_x - prev[:, 3]) >= respawn_threshold_bins
    pipe0_gap = torch.where(pipe0_respawn, logits["pipe0_gap"].argmax(dim=-1), prev[:, 2])
    pipe1_gap = torch.where(pipe1_respawn, logits["pipe1_gap"].argmax(dim=-1), prev[:, 4])
    action = target_action if target_action is not None else logits["action"].argmax(dim=-1)
    return torch.stack(
        (
            bird_y,
            pipe0_x,
            pipe0_gap,
            pipe1_x,
            pipe1_gap,
            (predicted_respawn.bool() | pipe0_respawn | pipe1_respawn).long(),
            (logits["done"] > 0).long(),
            action,
        ),
        dim=-1,
    )


@torch.no_grad()
def estimate_rollout_metrics(
    model: TokenizedARWorldModel,
    loader: DataLoader,
    device: torch.device,
    eval_iters: int,
    horizons: list[int],
    action_mode: str,
) -> dict[str, float]:
    if not horizons:
        return {}
    model.eval()
    max_horizon = max(horizons)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    iterator = iter(loader)
    for _ in range(eval_iters):
        try:
            context, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            context, targets = next(iterator)
        context = context.to(device)
        targets = targets.to(device)
        generated = context
        for step_idx in range(max_horizon):
            logits = model(generated[:, -model.config.history_size :])
            target = targets[:, step_idx]
            forced_action = target[:, 7] if action_mode == "teacher" else None
            pred = next_frame_prediction(logits, generated[:, -1], forced_action, model.config.respawn_threshold_bins)
            generated = torch.cat((generated, pred.unsqueeze(1)), dim=1)
            horizon = step_idx + 1
            if horizon not in horizons:
                continue
            prefix = f"rollout{horizon}"
            values = {
                f"{prefix}_bird_y_bin_mae": (pred[:, 0] - target[:, 0]).abs().float().mean(),
                f"{prefix}_pipe_x_bin_mae": 0.5
                * (
                    (pred[:, 1] - target[:, 1]).abs().float().mean()
                    + (pred[:, 3] - target[:, 3]).abs().float().mean()
                ),
                f"{prefix}_respawn_acc": (pred[:, 5] == target[:, 5]).float().mean(),
                f"{prefix}_done_acc": (pred[:, 6] == target[:, 6]).float().mean(),
                f"{prefix}_action_acc": (logits["action"].argmax(dim=-1) == target[:, 7]).float().mean(),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
                counts[key] = counts.get(key, 0) + 1
    model.train()
    return {key: totals[key] / max(1, counts[key]) for key in totals}


def rollout_step_loss(
    model: TokenizedARWorldModel,
    logits: dict[str, torch.Tensor],
    true_prev: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
    numeric_sigma: float,
    numeric_window: int,
) -> torch.Tensor:
    pipe0_respawn, pipe1_respawn = pipe_respawn_masks(true_prev, target, model.config)
    bird_y_loss = geometry_aware_ce(logits["bird_y"], target[:, 0], numeric_sigma, numeric_window).mean()
    pipe0_x_loss = geometry_aware_ce(logits["pipe0_x"], target[:, 1], numeric_sigma, numeric_window).mean()
    pipe1_x_loss = geometry_aware_ce(logits["pipe1_x"], target[:, 3], numeric_sigma, numeric_window).mean()
    pipe0_gap_loss = masked_mean(F.cross_entropy(logits["pipe0_gap"], target[:, 2], reduction="none"), pipe0_respawn)
    pipe1_gap_loss = masked_mean(F.cross_entropy(logits["pipe1_gap"], target[:, 4], reduction="none"), pipe1_respawn)
    respawn_loss = F.binary_cross_entropy_with_logits(logits["respawn"], target[:, 5].float())
    done_loss = F.binary_cross_entropy_with_logits(logits["done"], target[:, 6].float())
    action_loss = F.cross_entropy(logits["action"], target[:, 7])
    return (
        weights.bird_y * bird_y_loss
        + weights.pipe_x * (pipe0_x_loss + pipe1_x_loss)
        + weights.gap_respawn * (pipe0_gap_loss + pipe1_gap_loss)
        + weights.respawn * respawn_loss
        + weights.done * done_loss
        + weights.action * action_loss
    )


def rollout_training_loss(
    model: TokenizedARWorldModel,
    batch,
    device: torch.device,
    horizons: list[int],
    action_mode: str,
    weights: LossWeights,
    numeric_sigma: float,
    numeric_window: int,
) -> torch.Tensor:
    if not horizons:
        raise ValueError("rollout_training_loss requires at least one horizon")
    context, targets = [item.to(device) for item in batch]
    generated = context
    losses = []
    horizon_set = set(horizons)
    max_horizon = max(horizons)
    for step_idx in range(max_horizon):
        logits = model(generated[:, -model.config.history_size :])
        target = targets[:, step_idx]
        true_prev = context[:, -1] if step_idx == 0 else targets[:, step_idx - 1]
        if step_idx + 1 in horizon_set:
            losses.append(rollout_step_loss(model, logits, true_prev, target, weights, numeric_sigma, numeric_window))
        forced_action = target[:, 7] if action_mode == "teacher" else None
        pred = next_frame_prediction(logits, generated[:, -1], forced_action, model.config.respawn_threshold_bins)
        generated = torch.cat((generated, pred.detach().unsqueeze(1)), dim=1)
    return torch.stack(losses).mean()


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    keys = (
        "loss",
        "bird_y_loss",
        "pipe0_x_loss",
        "pipe1_x_loss",
        "pipe0_gap_loss",
        "pipe1_gap_loss",
        "respawn_loss",
        "done_loss",
        "action_loss",
        "bird_y_bin_mae",
        "pipe_x_bin_mae",
        "respawn_acc",
        "done_acc",
        "action_acc",
    )
    def rollout_sort_key(key: str) -> tuple[int, str]:
        horizon_text = key.removeprefix("rollout").split("_", 1)[0]
        return int(horizon_text), key

    rollout_keys = sorted((key for key in metrics if key.startswith("rollout")), key=rollout_sort_key)
    return " ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in (*keys, *rollout_keys) if key in metrics)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_payload(
    model: TokenizedARWorldModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    loss_weights: LossWeights,
    numeric_sigma: float,
    numeric_window: int,
    step: int,
    metrics: dict[str, float],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "tokenizer_config": asdict(tokenizer_config),
        "loss_weights": asdict(loss_weights),
        "numeric_soft_targets": {"sigma": numeric_sigma, "window": numeric_window},
        "frame_fields": ["bird_y_bin", "pipe0_x_bin", "pipe0_gap_bin", "pipe1_x_bin", "pipe1_gap_bin", "respawn", "done", "action"],
        "transition": "reference_model_v1",
        "step": step,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/flappy_lm_2pipes_ppo.jsonl")
    parser.add_argument("--out-dir", type=str, default="checkpoints/reference-model")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=64)
    parser.add_argument("--bird-y-bins", type=int, default=128)
    parser.add_argument("--pipe-x-bins", type=int, default=128)
    parser.add_argument("--pipe-gap-bins", type=int, default=96)
    parser.add_argument("--respawn-threshold-bins", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=25)
    parser.add_argument("--rollout-eval-horizons", type=str, default="8,16,32")
    parser.add_argument("--rollout-eval-iters", type=int, default=10)
    parser.add_argument("--rollout-eval-actions", choices=("teacher", "predicted"), default="teacher")
    parser.add_argument("--rollout-train-horizons", type=str, default="8,16,32")
    parser.add_argument("--rollout-train-weight", type=float, default=1.0)
    parser.add_argument("--rollout-train-actions", choices=("teacher", "predicted"), default="teacher")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--numeric-soft-target-sigma", type=float, default=1.5)
    parser.add_argument("--numeric-soft-target-window", type=int, default=5)
    parser.add_argument("--bird-y-weight", type=float, default=1.0)
    parser.add_argument("--pipe-x-weight", type=float, default=1.0)
    parser.add_argument("--gap-respawn-weight", type=float, default=1.0)
    parser.add_argument("--respawn-weight", type=float, default=2.0)
    parser.add_argument("--done-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    tokenizer_config = TokenizerConfig(
        bird_y_bins=args.bird_y_bins,
        pipe_x_bins=args.pipe_x_bins,
        pipe_gap_bins=args.pipe_gap_bins,
        respawn_threshold_bins=args.respawn_threshold_bins,
    )
    loss_weights = LossWeights(
        bird_y=args.bird_y_weight,
        pipe_x=args.pipe_x_weight,
        gap_respawn=args.gap_respawn_weight,
        respawn=args.respawn_weight,
        done=args.done_weight,
        action=args.action_weight,
    )
    episodes = load_episodes(Path(args.data), tokenizer_config, args.max_episodes)
    train_episodes, val_episodes = split_episodes(episodes)
    train_ds = FrameTransitionDataset(train_episodes, args.history_size)
    val_ds = FrameTransitionDataset(val_episodes, args.history_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    rollout_train_horizons = parse_rollout_horizons(args.rollout_train_horizons)
    train_rollout_loader = None
    if args.rollout_train_weight > 0.0 and rollout_train_horizons:
        train_rollout_ds = FrameRolloutDataset(train_episodes, args.history_size, max(rollout_train_horizons))
        train_rollout_loader = DataLoader(train_rollout_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    rollout_horizons = parse_rollout_horizons(args.rollout_eval_horizons)
    val_rollout_loader = None
    if rollout_horizons:
        val_rollout_ds = FrameRolloutDataset(val_episodes, args.history_size, max(rollout_horizons))
        val_rollout_loader = DataLoader(val_rollout_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model_config = ModelConfig(
        history_size=args.history_size,
        bird_y_bins=args.bird_y_bins,
        pipe_x_bins=args.pipe_x_bins,
        pipe_gap_bins=args.pipe_gap_bins,
        respawn_threshold_bins=args.respawn_threshold_bins,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = TokenizedARWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"device={device} episodes={len(episodes)} train_transitions={len(train_ds)} "
        f"val_transitions={len(val_ds)} params={parameter_count(model):,}",
        flush=True,
    )
    print(f"tokenizer_config={asdict(tokenizer_config)}", flush=True)
    print(f"loss_weights={asdict(loss_weights)}", flush=True)
    print(
        f"rollout_eval_horizons={rollout_horizons} rollout_eval_actions={args.rollout_eval_actions}",
        flush=True,
    )
    print(
        f"rollout_train_horizons={rollout_train_horizons} rollout_train_weight={args.rollout_train_weight} "
        f"rollout_train_actions={args.rollout_train_actions}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    train_iter = iter(train_loader)
    train_rollout_iter = iter(train_rollout_loader) if train_rollout_loader is not None else None
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            train_metrics = estimate_metrics(
                model,
                train_loader,
                device,
                min(args.eval_iters, max(1, len(train_loader))),
                loss_weights,
                args.numeric_soft_target_sigma,
                args.numeric_soft_target_window,
            )
            val_metrics = estimate_metrics(
                model,
                val_loader,
                device,
                min(args.eval_iters, max(1, len(val_loader))),
                loss_weights,
                args.numeric_soft_target_sigma,
                args.numeric_soft_target_window,
            )
            if val_rollout_loader is not None:
                val_metrics.update(
                    estimate_rollout_metrics(
                        model,
                        val_rollout_loader,
                        device,
                        min(args.rollout_eval_iters, max(1, len(val_rollout_loader))),
                        rollout_horizons,
                        args.rollout_eval_actions,
                    )
                )
            print(f"step={step} {format_metrics('train', train_metrics)} {format_metrics('val', val_metrics)}", flush=True)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        model_config,
                        tokenizer_config,
                        loss_weights,
                        args.numeric_soft_target_sigma,
                        args.numeric_soft_target_window,
                        step,
                        val_metrics,
                    ),
                    out_dir / "best.pt",
                )
        if step == args.steps:
            continue
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        loss, _metrics = loss_and_metrics(
            model,
            batch,
            device,
            loss_weights,
            args.numeric_soft_target_sigma,
            args.numeric_soft_target_window,
        )
        rollout_loss = loss.new_tensor(0.0)
        if train_rollout_loader is not None and train_rollout_iter is not None:
            try:
                rollout_batch = next(train_rollout_iter)
            except StopIteration:
                train_rollout_iter = iter(train_rollout_loader)
                rollout_batch = next(train_rollout_iter)
            rollout_loss = rollout_training_loss(
                model,
                rollout_batch,
                device,
                rollout_train_horizons,
                args.rollout_train_actions,
                loss_weights,
                args.numeric_soft_target_sigma,
                args.numeric_soft_target_window,
            )
            loss = loss + args.rollout_train_weight * rollout_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final_metrics = estimate_metrics(
        model,
        val_loader,
        device,
        min(args.eval_iters, max(1, len(val_loader))),
        loss_weights,
        args.numeric_soft_target_sigma,
        args.numeric_soft_target_window,
    )
    if val_rollout_loader is not None:
        final_metrics.update(
            estimate_rollout_metrics(
                model,
                val_rollout_loader,
                device,
                min(args.rollout_eval_iters, max(1, len(val_rollout_loader))),
                rollout_horizons,
                args.rollout_eval_actions,
            )
        )
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            model_config,
            tokenizer_config,
            loss_weights,
            args.numeric_soft_target_sigma,
            args.numeric_soft_target_window,
            args.steps,
            final_metrics,
        ),
        out_dir / "last.pt",
    )
    print(f"saved_best={out_dir / 'best.pt'} saved_last={out_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
