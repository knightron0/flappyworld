"""Print a naive token stream from a checkpoint to the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.core import FlatFrameLM, ModelConfig


DEFAULT_PROMPT = [
    "<bos>",
    "bird_y_061",
    "pipe0_present_1",
    "pipe0_x_127",
    "pipe0_gap_039",
    "pipe1_present_0",
    "pipe1_x_hidden",
    "pipe1_gap_hidden",
    "respawn_0",
    "done_0",
    "action_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--prompt-tokens", nargs="*", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = FlatFrameLM(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    vocab = {str(token): int(idx) for token, idx in ckpt["vocab"].items()}
    id_to_token = {int(key): str(value) for key, value in ckpt["id_to_token"].items()}
    ids = [vocab[token] for token in args.prompt_tokens]

    print("prompt:", " ".join(args.prompt_tokens), flush=True)
    generated: list[str] = []

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            trimmed_ids = ids[-384:] if len(ids) > 384 else ids
            input_ids = torch.tensor(trimmed_ids, dtype=torch.long).unsqueeze(0)
            position_ids = torch.arange(len(trimmed_ids), dtype=torch.long).unsqueeze(0)
       
            logits = model(input_ids, position_ids)
            next_id = int(logits[0, -1].argmax().item())
            next_token = id_to_token[next_id]
            print(next_token, flush=True)
            generated.append(next_token)
            ids.append(next_id)

    print("generated:", " ".join(generated), flush=True)


if __name__ == "__main__":
    main()
