"""Train a flat language model over compact tokenized Flappy frames.

This serializes frames into a single next-token stream:

    bird_y_... pipe0_x_... pipe0_gap_... pipe1_x_... pipe1_gap_...
            respawn_... done_... action_...

Every prediction comes from one vocabulary softmax.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.shared import FrameTokens, TokenizerConfig, load_episodes


CacheOutput = Literal["full", "delta"]


FRAME_FIELDS = (
    "bird_y",
    "pipe0_x",
    "pipe0_gap",
    "pipe1_x",
    "pipe1_gap",
    "respawn",
    "done",
    "action",
)
FRAME_FIELDS_1PIPE = (
    "bird_y",
    "pipe0_x",
    "pipe0_gap",
    "respawn",
    "done",
    "action",
)
NUMERIC_SOFT_FAMILIES = {"bird_y", "pipe0_x", "pipe1_x"}


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    max_position: int = 65536
    rope_theta: float = 10000.0
    position_scheme: str = "global_rope"
    mlp_mult: int = 4


@dataclass
class LossWeights:
    numeric: float
    gap_respawn: float
    event: float
    action: float
    structure: float
    death: float
    done_positive: float


@dataclass
class NumericSoftTargets:
    sigma: float
    window: int
    probs: torch.Tensor


class TokenDataset(Dataset):
    def __init__(self, ids: torch.Tensor, weights: torch.Tensor, positions: torch.Tensor, block_size: int):
        self.ids = ids
        self.weights = weights
        self.positions = positions
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, self.ids.numel() - self.block_size)

    def __getitem__(self, idx: int):
        sl = slice(idx, idx + self.block_size + 1)
        chunk = self.ids[sl]
        weight_chunk = self.weights[sl]
        pos_chunk = self.positions[sl]
        return chunk[:-1], chunk[1:], weight_chunk[:-1], weight_chunk[1:], pos_chunk[:-1], pos_chunk[1:]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (batch, heads, seq, head_dim); position_ids: (batch, seq)
    cos = cos.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1).unsqueeze(1)
    sin = sin.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


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
        self.rotary = RotaryEmbedding(self.head_dim, config.max_position, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        cache_output: CacheOutput = "full",
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        new_k, new_v = k, v
        if layer_cache is not None:
            past_k, past_v = layer_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        is_causal = layer_cache is None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = self.proj(y.transpose(1, 2).contiguous().view(batch, time, channels))
        if cache_output not in ("full", "delta"):
            raise ValueError(f"unknown cache_output={cache_output!r}")
        if not use_cache:
            new_cache = None
        elif cache_output == "delta":
            new_cache = (new_k, new_v)
        else:
            new_cache = (k, v)
        return out, new_cache


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        hidden_size = config.mlp_mult * config.n_embd
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.n_embd),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        cache_output: CacheOutput = "full",
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_cache = self.attn(self.ln_1(x), position_ids, layer_cache, use_cache, cache_output)
        x = x + attn_out
        return x + self.mlp(self.ln_2(x)), new_cache


class FlatFrameLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        cache_output: CacheOutput = "full",
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch, time = idx.shape
        if time > self.config.block_size and past_key_values is None:
            raise ValueError("sequence length exceeds block_size")
        if position_ids is None:
            position_ids = torch.arange(time, device=idx.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        x = self.drop(self.token_emb(idx))
        next_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if past_key_values is None else past_key_values[layer_idx]
            x, new_cache = block(x, position_ids, layer_cache, use_cache, cache_output)
            if use_cache:
                if new_cache is None:
                    raise RuntimeError("use_cache=True but attention returned no cache")
                next_caches.append(new_cache)
        logits = self.ln_f(x) @ self.lm_head.weight.T
        if use_cache:
            return logits, next_caches
        return logits


def token_family(token: str) -> str | None:
    if token in {"<bos>", "<eos>", "<DEATH>"}:
        return "structure"
    if token.startswith("bird_y_"):
        return "bird_y"
    if token.startswith("pipe0_x_"):
        return "pipe0_x"
    if token.startswith("pipe1_x_"):
        return "pipe1_x"
    if token.startswith("pipe0_gap_"):
        return "pipe0_gap"
    if token.startswith("pipe1_gap_"):
        return "pipe1_gap"
    if token.startswith(("respawn_", "done_")):
        return "event"
    if token.startswith("action_"):
        return "action"
    return None


def numeric_token_parts(token: str) -> tuple[str, int] | None:
    try:
        family, value = token.rsplit("_", 1)
        return family, int(value)
    except ValueError:
        return None


def frame_tokens(frame: FrameTokens, pipe_mode: int) -> list[tuple[str, str]]:
    tokens = [
        ("bird_y", f"bird_y_{frame.bird_y:03d}"),
        ("pipe0_x", f"pipe0_x_{frame.pipe0_x:03d}"),
        ("pipe0_gap", f"pipe0_gap_{frame.pipe0_gap:03d}"),
    ]
    if pipe_mode == 2:
        tokens.extend(
            [
                ("pipe1_x", f"pipe1_x_{frame.pipe1_x:03d}"),
                ("pipe1_gap", f"pipe1_gap_{frame.pipe1_gap:03d}"),
            ]
        )
    tokens.extend(
        [
        ("event", f"respawn_{frame.respawn}"),
        ("event", f"done_{frame.done}"),
        ("action", f"action_{frame.action}"),
        ]
    )
    return tokens


def build_vocab(cfg: TokenizerConfig, pipe_mode: int) -> tuple[dict[str, int], dict[int, str]]:
    tokens = ["<pad>", "<bos>", "<eos>", "<DEATH>"]
    tokens.extend(f"bird_y_{idx:03d}" for idx in range(cfg.bird_y_bins))
    tokens.extend(f"pipe0_x_{idx:03d}" for idx in range(cfg.pipe_x_bins))
    tokens.extend(f"pipe0_gap_{idx:03d}" for idx in range(cfg.pipe_gap_bins))
    if pipe_mode == 2:
        tokens.extend(f"pipe1_x_{idx:03d}" for idx in range(cfg.pipe_x_bins))
        tokens.extend(f"pipe1_gap_{idx:03d}" for idx in range(cfg.pipe_gap_bins))
    tokens.extend(["respawn_0", "respawn_1", "done_0", "done_1", "action_0", "action_1"])
    vocab = {token: idx for idx, token in enumerate(tokens)}
    return vocab, {idx: token for token, idx in vocab.items()}


def field_loss_weight(field: str, pipe0_respawn: bool, pipe1_respawn: bool, weights: LossWeights) -> float:
    if field in NUMERIC_SOFT_FAMILIES:
        return weights.numeric
    if field == "pipe0_gap":
        return weights.gap_respawn
    if field == "pipe1_gap":
        return weights.gap_respawn
    if field == "event":
        return weights.event
    if field == "action":
        return weights.action
    if field == "structure":
        return weights.structure
    raise ValueError(f"Unknown token field: {field}")


def token_loss_weight(token: str, frame_has_respawn: bool, weights: LossWeights) -> float:
    if token in {"<bos>", "<eos>"}:
        return weights.structure
    if token == "<DEATH>":
        return weights.death
    if token == "done_1":
        return weights.done_positive
    if token.startswith(("bird_y_", "pipe0_x_", "pipe1_x_")):
        return weights.numeric
    if token.startswith(("pipe0_gap_", "pipe1_gap_")):
        return weights.gap_respawn
    if token.endswith("_hidden"):
        return weights.structure
    if token.startswith(("pipe0_present_", "pipe1_present_", "respawn_", "done_")):
        return weights.event
    if token.startswith("action_"):
        return weights.action
    return weights.structure


def visible_record_weights(tokens: list[str], weights: LossWeights) -> list[float]:
    return [token_loss_weight(token, False, weights) for token in tokens]


def load_visible_token_records(path: Path, max_records: int | None, weights: LossWeights) -> tuple[list[list[str]], list[list[float]], TokenizerConfig]:
    records: list[list[str]] = []
    record_weights: list[list[float]] = []
    tokenizer_config = None
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_records is not None and line_idx >= max_records:
                break
            record = json.loads(line)
            if record.get("format") != "visible_pipe_flat_lm_v1":
                raise ValueError(f"Expected visible_pipe_flat_lm_v1, got {record.get('format')!r}")
            tokens = [token for token in record["tokens"] if token not in {"<frame>", "</frame>"}]
            if tokenizer_config is None:
                cfg = record["visible_pipe_tokenizer"]
                tokenizer_config = TokenizerConfig(
                    bird_y_bins=int(cfg["bird_y_bins"]),
                    pipe_x_bins=int(cfg["pipe_x_bins"]),
                    pipe_gap_bins=int(cfg["pipe_gap_bins"]),
                    respawn_threshold_bins=int(cfg["respawn_threshold_bins"]),
                )
            records.append(tokens)
            record_weights.append(visible_record_weights(tokens, weights))
    if tokenizer_config is None:
        raise ValueError(f"No visible-pipe token records found in {path}")
    return records, record_weights, tokenizer_config


def episode_position_ids(length: int) -> list[int]:
    return list(range(length))


def flatten_token_records(
    records: list[list[str]],
    record_weights: list[list[float]],
    vocab: dict[str, int],
) -> tuple[list[int], list[float], list[int]]:
    ids: list[int] = []
    weights: list[float] = []
    positions: list[int] = []
    for tokens, token_weights in zip(records, record_weights, strict=True):
        if len(tokens) != len(token_weights):
            raise ValueError("Token and weight lengths differ")
        ids.extend(vocab[token] for token in tokens)
        weights.extend(token_weights)
        positions.extend(episode_position_ids(len(tokens)))
    return ids, weights, positions


def serialize_episodes(
    episodes,
    vocab: dict[str, int],
    weights: LossWeights,
    respawn_threshold_bins: int,
    pipe_mode: int,
) -> tuple[list[int], list[float], list[int]]:
    ids: list[int] = []
    loss_weights: list[float] = []
    positions: list[int] = []
    for episode in episodes:
        episode_pos = 0
        ids.append(vocab["<bos>"])
        loss_weights.append(weights.structure)
        positions.append(episode_pos)
        episode_pos += 1
        prev = None
        for frame in episode.frames:
            pipe0_respawn = prev is not None and frame.pipe0_x - prev.pipe0_x >= respawn_threshold_bins
            pipe1_respawn = pipe_mode == 2 and prev is not None and frame.pipe1_x - prev.pipe1_x >= respawn_threshold_bins
            for field, token in frame_tokens(frame, pipe_mode):
                ids.append(vocab[token])
                loss_weights.append(field_loss_weight(field, pipe0_respawn, pipe1_respawn, weights))
                positions.append(episode_pos)
                episode_pos += 1
            prev = frame
        if episode.final_done and episode.frames:
            final = episode.frames[-1]
            death = FrameTokens(
                bird_y=final.bird_y,
                pipe0_x=final.pipe0_x,
                pipe0_gap=final.pipe0_gap,
                pipe1_x=final.pipe1_x,
                pipe1_gap=final.pipe1_gap,
                respawn=0,
                done=1,
                action=final.action,
            )
            for field, token in frame_tokens(death, pipe_mode):
                ids.append(vocab[token])
                if token == "done_1":
                    loss_weights.append(weights.done_positive)
                elif field in {"event", "structure"}:
                    loss_weights.append(weights.event)
                else:
                    loss_weights.append(weights.numeric)
                positions.append(episode_pos)
                episode_pos += 1
            ids.append(vocab["<DEATH>"])
            loss_weights.append(weights.death)
            positions.append(episode_pos)
            episode_pos += 1
        ids.append(vocab["<eos>"])
        loss_weights.append(weights.structure)
        positions.append(episode_pos)
    return ids, loss_weights, positions


def build_numeric_soft_targets(vocab: dict[str, int], sigma: float, window: int) -> NumericSoftTargets | None:
    if sigma <= 0.0:
        return None
    if window <= 0:
        window = max(1, math.ceil(3.0 * sigma))
    probs = torch.eye(len(vocab), dtype=torch.float32)
    by_family: dict[str, list[tuple[int, int]]] = {}
    for token, token_id in vocab.items():
        parts = numeric_token_parts(token)
        if parts is None:
            continue
        family, value = parts
        if family not in NUMERIC_SOFT_FAMILIES:
            continue
        by_family.setdefault(family, []).append((token_id, value))

    for items in by_family.values():
        ids = torch.tensor([token_id for token_id, _value in items], dtype=torch.long)
        values = torch.tensor([value for _token_id, value in items], dtype=torch.float32)
        for target_id, target_value in items:
            distances = (values - float(target_value)).abs()
            keep = distances <= float(window)
            local_ids = ids[keep]
            local_probs = torch.exp(-0.5 * (distances[keep] / sigma).pow(2))
            local_probs = local_probs / local_probs.sum().clamp_min(1e-8)
            probs[target_id].zero_()
            probs[target_id, local_ids] = local_probs
    return NumericSoftTargets(sigma=sigma, window=window, probs=probs)


def compute_token_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    soft_targets: NumericSoftTargets | None,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if soft_targets is None:
        return F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(targets)
    probs = soft_targets.probs.to(flat_logits.device, dtype=flat_logits.dtype).index_select(0, flat_targets)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).view_as(targets)


def batch_loss(
    model: FlatFrameLM,
    batch,
    device: torch.device,
    soft_targets: NumericSoftTargets | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, y, _x_weights, weights, position_ids, _target_position_ids = [item.to(device) for item in batch]
    logits = model(x, position_ids)
    losses = compute_token_losses(logits, y, soft_targets)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, {"loss": float(loss.item())}


def parse_rollout_horizons(value: str) -> list[int]:
    horizons: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError("--rollout-train-horizons ranges must be positive and increasing")
            horizons.update(range(start, end + 1))
        else:
            horizon = int(item)
            if horizon <= 0:
                raise ValueError("--rollout-train-horizons must contain positive integers")
            horizons.add(horizon)
    return sorted(horizons)


def rollout_training_loss(
    model: FlatFrameLM,
    batch,
    device: torch.device,
    soft_targets: NumericSoftTargets | None,
    horizons: list[int],
) -> torch.Tensor:
    if not horizons:
        raise ValueError("rollout_training_loss requires at least one horizon")
    x, _y, x_weights, _y_weights, position_ids, _target_position_ids = [item.to(device) for item in batch]
    max_horizon = max(horizons)
    if x.size(1) <= max_horizon:
        return x.new_tensor(0.0, dtype=torch.float32)

    context_len = x.size(1) - max_horizon
    generated = x[:, :context_len]
    generated_positions = position_ids[:, :context_len]
    horizon_set = set(horizons)
    losses: list[torch.Tensor] = []

    for step_idx in range(max_horizon):
        logits = model(generated, generated_positions)
        next_logits = logits[:, -1, :]
        target_idx = context_len + step_idx
        target = x[:, target_idx]
        if step_idx + 1 in horizon_set:
            token_losses = compute_token_losses(next_logits.unsqueeze(1), target.unsqueeze(1), soft_targets).squeeze(1)
            target_weights = x_weights[:, target_idx]
            losses.append((token_losses * target_weights).sum() / target_weights.sum().clamp_min(1e-8))
        pred = next_logits.argmax(dim=-1)
        generated = torch.cat((generated, pred.detach().unsqueeze(1)), dim=1)
        generated_positions = torch.cat((generated_positions, position_ids[:, target_idx].unsqueeze(1)), dim=1)
    return torch.stack(losses).mean()


@torch.no_grad()
def estimate_loss(
    model: FlatFrameLM,
    loader: DataLoader,
    device: torch.device,
    eval_iters: int,
    soft_targets: NumericSoftTargets | None,
) -> dict[str, float]:
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(eval_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loss, _metrics = batch_loss(model, batch, device, soft_targets)
        losses.append(float(loss.item()))
    model.train()
    return {"loss": sum(losses) / max(1, len(losses))}


def split_episodes(episodes, train_frac: float = 0.9):
    split = max(1, int(len(episodes) * train_frac))
    if split >= len(episodes):
        return episodes, episodes
    return episodes[:split], episodes[split:]


def split_records(records, record_weights, train_frac: float = 0.9):
    split = max(1, int(len(records) * train_frac))
    if split >= len(records):
        return records, record_weights, records, record_weights
    return records[:split], record_weights[:split], records[split:], record_weights[split:]


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_payload(
    model: FlatFrameLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    loss_weights: LossWeights,
    soft_targets: NumericSoftTargets | None,
    pipe_mode: int,
    step: int,
    metrics: dict[str, float],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "tokenizer_config": asdict(tokenizer_config),
        "vocab": vocab,
        "id_to_token": id_to_token,
        "loss_weights": asdict(loss_weights),
        "numeric_soft_targets": None if soft_targets is None else {"sigma": soft_targets.sigma, "window": soft_targets.window},
        "frame_fields": FRAME_FIELDS_1PIPE if pipe_mode == 1 else FRAME_FIELDS,
        "pipe_mode": pipe_mode,
        "transition": "browser_model_v1_global_rope",
        "step": step,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/flappy_lm_2pipes_ppo.jsonl")
    parser.add_argument("--data-format", choices=("auto", "raw", "visible_tokens"), default="auto")
    parser.add_argument("--out-dir", type=str, default="checkpoints/model")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=640)
    parser.add_argument("--bird-y-bins", type=int, default=128)
    parser.add_argument("--pipe-x-bins", type=int, default=128)
    parser.add_argument("--pipe-gap-bins", type=int, default=96)
    parser.add_argument("--pipe-mode", type=int, choices=(1, 2), default=2)
    parser.add_argument("--respawn-threshold-bins", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--numeric-soft-target-sigma", type=float, default=1.5)
    parser.add_argument("--numeric-soft-target-window", type=int, default=5)
    parser.add_argument("--numeric-weight", type=float, default=1.0)
    parser.add_argument("--gap-respawn-weight", type=float, default=1.0)
    parser.add_argument("--event-weight", type=float, default=2.0)
    parser.add_argument("--action-weight", type=float, default=0.2)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--death-weight", type=float, default=15.0)
    parser.add_argument("--done-positive-weight", type=float, default=15.0)
    parser.add_argument("--rollout-train-horizons", type=str, default="4-8")
    parser.add_argument("--rollout-train-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-position", type=int, default=65536, help="RoPE table size; must exceed max per-episode token count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    loss_weights = LossWeights(
        numeric=args.numeric_weight,
        gap_respawn=args.gap_respawn_weight,
        event=args.event_weight,
        action=args.action_weight,
        structure=args.structure_weight,
        death=args.death_weight,
        done_positive=args.done_positive_weight,
    )
    data_path = Path(args.data)
    data_format = args.data_format
    if data_format == "auto":
        with data_path.open("r", encoding="utf-8") as f:
            first_record = json.loads(next(f))
        data_format = "visible_tokens" if first_record.get("format") == "visible_pipe_flat_lm_v1" else "raw"

    if data_format == "visible_tokens":
        records, record_weights, tokenizer_config = load_visible_token_records(data_path, args.max_episodes, loss_weights)
        train_records, train_record_weights, val_records, val_record_weights = split_records(records, record_weights)
        all_tokens = sorted({token for record in records for token in record})
        vocab = {token: idx for idx, token in enumerate(all_tokens)}
        id_to_token = {idx: token for token, idx in vocab.items()}
        train_ids, train_weights, train_positions = flatten_token_records(train_records, train_record_weights, vocab)
        val_ids, val_weights, val_positions = flatten_token_records(val_records, val_record_weights, vocab)
        episode_count = len(records)
        pipe_mode = 2
    else:
        tokenizer_config = TokenizerConfig(
            bird_y_bins=args.bird_y_bins,
            pipe_x_bins=args.pipe_x_bins,
            pipe_gap_bins=args.pipe_gap_bins,
            respawn_threshold_bins=args.respawn_threshold_bins,
        )
        episodes = load_episodes(data_path, tokenizer_config, args.max_episodes)
        train_episodes, val_episodes = split_episodes(episodes)
        vocab, id_to_token = build_vocab(tokenizer_config, args.pipe_mode)
        train_ids, train_weights, train_positions = serialize_episodes(
            train_episodes, vocab, loss_weights, tokenizer_config.respawn_threshold_bins, args.pipe_mode
        )
        val_ids, val_weights, val_positions = serialize_episodes(
            val_episodes, vocab, loss_weights, tokenizer_config.respawn_threshold_bins, args.pipe_mode
        )
        episode_count = len(episodes)
        pipe_mode = args.pipe_mode
    train_ds = TokenDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(train_weights, dtype=torch.float32),
        torch.tensor(train_positions, dtype=torch.long),
        args.block_size,
    )
    val_ds = TokenDataset(
        torch.tensor(val_ids, dtype=torch.long),
        torch.tensor(val_weights, dtype=torch.float32),
        torch.tensor(val_positions, dtype=torch.long),
        args.block_size,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    observed_max_position = max(int(train_ds.positions.max()), int(val_ds.positions.max()))
    max_position = max(args.max_position, observed_max_position + 1)
    if max_position > args.max_position:
        print(
            f"max_position_bump={max_position} reason=observed_max_token_position_{observed_max_position}",
            flush=True,
        )
    model_config = ModelConfig(
        vocab_size=len(vocab),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        max_position=max_position,
        mlp_mult=args.mlp_mult,
    )
    model = FlatFrameLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    soft_targets = build_numeric_soft_targets(vocab, args.numeric_soft_target_sigma, args.numeric_soft_target_window)
    rollout_train_horizons = parse_rollout_horizons(args.rollout_train_horizons)
    print(
        f"device={device} episodes={episode_count} tokens={len(train_ids) + len(val_ids)} "
        f"vocab={len(vocab)} params={parameter_count(model):,}",
        flush=True,
    )
    print(f"tokenizer_config={asdict(tokenizer_config)}", flush=True)
    print(f"loss_weights={asdict(loss_weights)}", flush=True)
    print(
        f"rollout_train_horizons={rollout_train_horizons} rollout_train_weight={args.rollout_train_weight}",
        flush=True,
    )
    print(
        f"data_format={data_format} pipe_mode={pipe_mode} position_scheme={model_config.position_scheme} "
        f"max_position={model_config.max_position} observed_max_position={observed_max_position}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    train_iter = iter(train_loader)
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            train_metrics = estimate_loss(model, train_loader, device, min(args.eval_iters, max(1, len(train_loader))), soft_targets)
            val_metrics = estimate_loss(model, val_loader, device, min(args.eval_iters, max(1, len(val_loader))), soft_targets)
            print(f"step={step} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}", flush=True)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        model_config,
                        tokenizer_config,
                        vocab,
                        id_to_token,
                        loss_weights,
                        soft_targets,
                        pipe_mode,
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
        loss, _metrics = batch_loss(model, batch, device, soft_targets)
        rollout_loss = loss.new_tensor(0.0)
        if args.rollout_train_weight > 0.0 and rollout_train_horizons:
            rollout_loss = rollout_training_loss(model, batch, device, soft_targets, rollout_train_horizons)
            loss = loss + args.rollout_train_weight * rollout_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_metrics = estimate_loss(model, val_loader, device, min(args.eval_iters, max(1, len(val_loader))), soft_targets)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            model_config,
            tokenizer_config,
            vocab,
            id_to_token,
            loss_weights,
            soft_targets,
            pipe_mode,
            args.steps,
            final_metrics,
        ),
        out_dir / "last.pt",
    )
    print(f"saved_best={out_dir / 'best.pt'} saved_last={out_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
