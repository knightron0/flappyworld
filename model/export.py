"""Serving-friendly flat LM inference with explicit KV cache state.

PyTorch reference API
---------------------
Use :class:`FlatLMInferenceEngine` for cache-friendly prefill/decode:

    engine = FlatLMInferenceEngine.from_checkpoint(path, device)
    logits, cache = engine.prefill(input_ids, position_ids)
    logits, cache = engine.step(token_id, position_id, cache)

ONNX export
-----------
Export two graphs that mirror the PyTorch API:

1. ``prefill``  - bootstrap context, returns logits + per-layer K/V
2. ``decode``   - single-token step with past K/V in/out

Run:

    python -m scripts.export_model export \\
        --checkpoint path/to/best.pt \\
        --out-dir path/to/onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model.shared import TokenizerConfig
from model.core import FlatFrameLM, ModelConfig


@dataclass
class KVCache:
    """Per-layer attention cache. Tensors are (batch, n_head, seq, head_dim)."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @property
    def n_layer(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        if not self.keys:
            return 0
        return int(self.keys[0].shape[2])

    @property
    def batch_size(self) -> int:
        if not self.keys:
            return 0
        return int(self.keys[0].shape[0])

    def truncate_left(self, drop: int) -> KVCache:
        if drop <= 0:
            return self
        return KVCache(
            keys=[k[:, :, drop:, :].contiguous() for k in self.keys],
            values=[v[:, :, drop:, :].contiguous() for v in self.values],
        )

    def append(self, delta: KVCache) -> KVCache:
        if self.n_layer != delta.n_layer:
            raise ValueError(f"cache layer mismatch: {self.n_layer} != {delta.n_layer}")
        return KVCache(
            keys=[
                torch.cat([key, new_key], dim=2)
                for key, new_key in zip(self.keys, delta.keys, strict=True)
            ],
            values=[
                torch.cat([value, new_value], dim=2)
                for value, new_value in zip(self.values, delta.values, strict=True)
            ],
        )

    def to_legacy(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self.keys, self.values, strict=True))

    @classmethod
    def from_legacy(cls, past_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> KVCache:
        keys, values = zip(*past_key_values, strict=True) if past_key_values else ([], [])
        return cls(keys=list(keys), values=list(values))

    @classmethod
    def empty(cls, n_layer: int, batch: int, n_head: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        shape = (batch, n_head, 0, head_dim)
        return cls(
            keys=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)],
            values=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)],
        )


def _as_batch_2d(ids: torch.Tensor) -> torch.Tensor:
    if ids.ndim == 0:
        return ids.view(1, 1)
    if ids.ndim == 1:
        return ids.view(1, -1)
    if ids.ndim == 2:
        return ids
    raise ValueError(f"expected rank 0/1/2 tensor, got shape {tuple(ids.shape)}")


def _as_position_batch(position_ids: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    pos = _as_batch_2d(position_ids).to(dtype=torch.long)
    if pos.shape[0] == 1 and batch > 1:
        pos = pos.expand(batch, -1)
    if pos.shape[0] != batch or pos.shape[1] != seq:
        raise ValueError(f"position_ids shape {tuple(pos.shape)} does not match input ({batch}, {seq})")
    return pos


class FlatLMInferenceEngine:
    """Cache-friendly inference wrapper around :class:`FlatFrameLM`."""

    def __init__(self, model: FlatFrameLM, device: torch.device | str):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def config(self) -> ModelConfig:
        return self.model.config

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: torch.device | str | None = None) -> FlatLMInferenceEngine:
        path = Path(checkpoint)
        resolved = torch.device(device) if device is not None else _default_device()
        ckpt = torch.load(path, map_location=resolved)
        model_config = ModelConfig(**ckpt["model_config"])
        model = FlatFrameLM(model_config)
        model.load_state_dict(ckpt["model"])
        return cls(model, resolved)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Run the model on ``input_ids`` with global ``position_ids``.

        ``cache=None`` performs a prefill pass and returns a full cache. Passing
        a cache asks the model for only the new token K/V tensors, then appends
        them to the provided cache.
        """
        input_ids = _as_batch_2d(input_ids).to(device=self.device, dtype=torch.long)
        batch, seq = input_ids.shape
        position_ids = _as_position_batch(position_ids, batch, seq).to(device=self.device)
        past = None if cache is None else cache.to_legacy()
        cache_output = "full" if cache is None else "delta"
        logits, past_out = self.model(
            input_ids,
            position_ids,
            past_key_values=past,
            use_cache=True,
            cache_output=cache_output,
        )
        next_cache = KVCache.from_legacy(past_out)
        if cache is not None:
            next_cache = cache.append(next_cache)
        return logits, next_cache

    def prefill(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, KVCache]:
        return self.forward(input_ids, position_ids, cache=None)

    def step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache,
    ) -> tuple[torch.Tensor, KVCache]:
        """Single (or few) token decode step with an existing cache."""
        return self.forward(input_ids, position_ids, cache=cache)

    @staticmethod
    def logits_at_last(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, :]


class OnnxFlatLMPrefill(nn.Module):
    """ONNX graph: context bootstrap with cache outputs."""

    def __init__(self, model: FlatFrameLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits, caches = self.model(input_ids, position_ids, use_cache=True)
        outputs: list[torch.Tensor] = [logits]
        for key, value in caches:
            outputs.extend([key, value])
        return tuple(outputs)


class OnnxFlatLMDecode(nn.Module):
    """ONNX graph: append tokens to an existing cache."""

    def __init__(self, model: FlatFrameLM):
        super().__init__()
        self.model = model
        self.n_layer = len(model.blocks)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, *past_flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(past_flat) != 2 * self.n_layer:
            raise ValueError(f"expected {2 * self.n_layer} past tensors, got {len(past_flat)}")
        past_key_values = [(past_flat[i], past_flat[i + 1]) for i in range(0, len(past_flat), 2)]
        logits, caches = self.model(
            input_ids,
            position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_output="delta",
        )
        outputs: list[torch.Tensor] = [logits]
        for key, value in caches:
            outputs.extend([key, value])
        return tuple(outputs)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _onnx_io_names(n_layer: int, *, decode: bool) -> tuple[list[str], list[str]]:
    inputs = ["input_ids", "position_ids"]
    if decode:
        for layer in range(n_layer):
            inputs.extend([f"past_key_{layer}", f"past_value_{layer}"])
    outputs = ["logits"]
    prefix = "new" if decode else "present"
    for layer in range(n_layer):
        outputs.extend([f"{prefix}_key_{layer}", f"{prefix}_value_{layer}"])
    return inputs, outputs


def export_onnx(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    opset: int = 17,
    prefill_seq: int = 128,
) -> tuple[Path, Path]:
    """Export prefill + decode ONNX graphs from a trained checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine = FlatLMInferenceEngine.from_checkpoint(checkpoint, device="cpu")
    model = engine.model
    n_layer = len(model.blocks)
    cfg = model.config

    prefill_wrapper = OnnxFlatLMPrefill(model).eval()
    decode_wrapper = OnnxFlatLMDecode(model).eval()

    prefill_ids = torch.zeros((1, prefill_seq), dtype=torch.long)
    prefill_pos = torch.arange(prefill_seq, dtype=torch.long).unsqueeze(0)
    prefill_path = out / "flat_lm_prefill.onnx"
    torch.onnx.export(
        prefill_wrapper,
        (prefill_ids, prefill_pos),
        prefill_path,
        input_names=["input_ids", "position_ids"],
        output_names=_onnx_io_names(n_layer, decode=False)[1],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            **{
                f"present_key_{i}": {0: "batch", 2: "seq"}
                for i in range(n_layer)
            },
            **{
                f"present_value_{i}": {0: "batch", 2: "seq"}
                for i in range(n_layer)
            },
        },
        opset_version=opset,
        dynamo=False,
    )

    decode_ids = torch.zeros((1, 1), dtype=torch.long)
    decode_pos = torch.tensor([[prefill_seq]], dtype=torch.long)
    empty_cache = KVCache.empty(
        n_layer=n_layer,
        batch=1,
        n_head=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    decode_args = (decode_ids, decode_pos, *empty_cache.keys, *empty_cache.values)
    decode_path = out / "flat_lm_decode.onnx"
    torch.onnx.export(
        decode_wrapper,
        decode_args,
        decode_path,
        input_names=_onnx_io_names(n_layer, decode=True)[0],
        output_names=_onnx_io_names(n_layer, decode=True)[1],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            **{
                f"past_key_{i}": {0: "batch", 2: "past_seq"}
                for i in range(n_layer)
            },
            **{
                f"past_value_{i}": {0: "batch", 2: "past_seq"}
                for i in range(n_layer)
            },
            **{
                f"new_key_{i}": {0: "batch", 2: "new_seq"}
                for i in range(n_layer)
            },
            **{
                f"new_value_{i}": {0: "batch", 2: "new_seq"}
                for i in range(n_layer)
            },
        },
        opset_version=opset,
        dynamo=False,
    )
    return prefill_path, decode_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flat LM serving utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export prefill/decode ONNX graphs")
    export.add_argument("--checkpoint", type=str, required=True)
    export.add_argument("--out-dir", type=str, required=True)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--prefill-seq", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export":
        prefill_path, decode_path = export_onnx(
            args.checkpoint,
            args.out_dir,
            opset=args.opset,
            prefill_seq=args.prefill_seq,
        )
        print(f"prefill_onnx={prefill_path}")
        print(f"decode_onnx={decode_path}")


if __name__ == "__main__":
    main()
