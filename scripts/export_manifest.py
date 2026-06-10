"""Export browser manifest.json from a flat LM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.play import visible_seed_frames_from_data
from model.player_base import seed_frames_from_data
from model.data import TokenizerConfig


def load_default_seed(
    path: Path | None,
    line: int,
    state_index: int,
    history_size: int,
    tokenizer_config: TokenizerConfig,
) -> dict | None:
    if path is None or not path.exists():
        return None
    visible = visible_seed_frames_from_data(path, line, state_index, history_size, tokenizer_config)
    if visible is not None:
        frames, present = visible
        return {
            "format": "visible_pipe_flat_lm_v1",
            "line": line,
            "state_index": state_index,
            "frames": [asdict(frame) for frame in frames],
            "pipe_present": [list(item) for item in present],
        }
    frames = seed_frames_from_data(path, tokenizer_config, line, state_index, history_size)
    return {
        "format": "episode_frames_v1",
        "line": line,
        "state_index": state_index,
        "frames": [asdict(frame) for frame in frames],
        "pipe_present": None,
    }


def _onnx_manifest(n_layer: int) -> dict:
    return {
        "prefill": "flat_lm_prefill.onnx",
        "decode": "flat_lm_decode.onnx",
        "decode_cache_output": "delta",
        "n_layer": n_layer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export flat LM manifest for browser player")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="web/public/model/manifest.json")
    parser.add_argument("--seed-data", type=str, default=None, help="Optional JSONL for default_seed")
    parser.add_argument("--seed-line", type=int, default=0)
    parser.add_argument("--seed-state-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_config = ckpt["model_config"]
    position_scheme = str(model_config.get("position_scheme", "local_pos_emb"))
    transition = str(ckpt.get("transition", ""))
    if position_scheme != "global_rope" and "global_rope" not in transition:
        raise SystemExit("Checkpoint must use global_rope position scheme.")
    vocab = ckpt["vocab"]
    id_to_token = {str(int(k)): v for k, v in ckpt["id_to_token"].items()}
    block_size = int(model_config["block_size"])
    n_layer = int(model_config["n_layer"])
    history_size = max(1, block_size // 10)
    tokenizer_config = TokenizerConfig(**ckpt["tokenizer_config"])
    default_seed = load_default_seed(
        Path(args.seed_data) if args.seed_data else None,
        args.seed_line,
        args.seed_state_index,
        history_size,
        tokenizer_config,
    )
    manifest = {
        "player_version": "global-rope-serve-v1",
        "model_cache_key": (
            f"{transition}-v{model_config['vocab_size']}-b{model_config['block_size']}"
            f"-l{model_config['n_layer']}-h{model_config['n_embd']}"
        ),
        "transition": transition,
        "position_scheme": position_scheme,
        "model_config": model_config,
        "tokenizer_config": ckpt["tokenizer_config"],
        "vocab": vocab,
        "id_to_token": id_to_token,
        "onnx": _onnx_manifest(n_layer),
        "history_size": history_size,
    }
    if default_seed is not None:
        manifest["default_seed"] = default_seed
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={out_path}")
    print(f"vocab_size={model_config['vocab_size']} n_layer={n_layer} block_size={block_size}")


if __name__ == "__main__":
    main()
