"""Interactively play the flat serialized token LM world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model.player_base import (
    PygamePlayer,
    StepResult,
    frame_to_render_state,
    seed_frames_from_data,
)
from model.export import FlatLMInferenceEngine, KVCache
from model.data import FrameTokens, ModelConfig as TypedModelConfig, TokenizerConfig
from model.core import FlatFrameLM, ModelConfig
from model.render import SCREEN_HEIGHT, SCREEN_WIDTH


torch: Any = None
PLAYER_VERSION = "global-rope-serve-v1"


def default_device() -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model/best.pt")
    parser.add_argument("--prompt-from-data", type=str, default="dataset/flappy_lm_2pipes_ppo.jsonl")
    parser.add_argument("--line", type=int, default=0)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--pipe-gap-px", type=int, default=100)
    parser.add_argument("--hide-guides", action="store_true")
    parser.add_argument("--trace-output", type=str, default="dataset/play_trace.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path, device: Any) -> tuple[FlatFrameLM, TokenizerConfig, dict[str, int], dict[int, str], str]:
    print(f"loading_checkpoint={path}", flush=True)
    ckpt = torch.load(path, map_location=device)
    model_config = ModelConfig(**ckpt["model_config"])
    position_scheme = str(ckpt.get("model_config", {}).get("position_scheme", "local_pos_emb"))
    transition = str(ckpt.get("transition", ""))
    if position_scheme != "global_rope" and "global_rope" not in transition:
        raise SystemExit(
            "Checkpoint uses legacy local position embeddings. Retrain with the updated "
            "scripts.train_model (global RoPE) before playing with KV cache."
        )
    tokenizer_config = TokenizerConfig(**ckpt["tokenizer_config"])
    print(f"building_model={model_config} position_scheme={position_scheme}", flush=True)
    model = FlatFrameLM(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    id_to_token = {int(key): value for key, value in ckpt["id_to_token"].items()}
    print("checkpoint_loaded=yes", flush=True)
    return model, tokenizer_config, ckpt["vocab"], id_to_token, position_scheme


def visible_seed_frames_from_data(
    path: Path,
    line: int,
    state_index: int,
    history_size: int,
    tokenizer_config: TokenizerConfig,
) -> tuple[list[FrameTokens], list[tuple[bool, bool]]] | None:
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            if idx != line:
                continue
            record = json.loads(raw)
            if record.get("format") != "visible_pipe_flat_lm_v1" or "visible_pipe_frames" not in record:
                return None
            frames = record["visible_pipe_frames"]
            if state_index >= len(frames):
                raise IndexError(f"--state-index {state_index} out of range; episode has {len(frames)} visible frames")
            start = max(0, state_index - history_size + 1)
            out_frames: list[FrameTokens] = []
            out_present: list[tuple[bool, bool]] = []
            for frame in frames[start : state_index + 1]:
                p0, p1 = frame["pipes"]
                p0_present = bool(p0["present"])
                p1_present = bool(p1["present"])
                out_frames.append(
                    FrameTokens(
                        bird_y=int(frame["bird_y"]),
                        pipe0_x=int(p0["x"]) if p0_present else tokenizer_config.pipe_x_bins - 1,
                        pipe0_gap=int(p0["gap"]) if p0_present else 0,
                        pipe1_x=int(p1["x"]) if p1_present else tokenizer_config.pipe_x_bins - 1,
                        pipe1_gap=int(p1["gap"]) if p1_present else 0,
                        respawn=int(frame["respawn"]),
                        done=int(frame["done"]),
                        action=int(frame["action"]),
                    )
                )
                out_present.append((p0_present, p1_present))
            return out_frames, out_present
    raise IndexError(f"{path} has no line {line}")


def parse_value_token(token: str) -> int:
    return int(token.rsplit("_", 1)[1])


def sample_from_logits(logits: Any) -> int:
    return int(logits.argmax(dim=-1).item())


class FlatLMStepper:
    def __init__(
        self,
        engine: FlatLMInferenceEngine,
        vocab: dict[str, int],
        id_to_token: dict[int, str],
        tokenizer_config: TokenizerConfig,
        seed_frames: list[FrameTokens],
        seed_present: list[tuple[bool, bool]] | None,
        device: Any,
    ):
        self.engine = engine
        self.vocab = vocab
        self.id_to_token = id_to_token
        self.tokenizer_config = tokenizer_config
        self.device = device
        self.initial_frames = [FrameTokens(**frame.__dict__) for frame in seed_frames]
        self.initial_present = list(seed_present) if seed_present is not None else None
        self.reset()

    @property
    def typed_config(self) -> TypedModelConfig:
        # Rendering only needs bin counts and respawn threshold.
        return TypedModelConfig(
            history_size=0,
            bird_y_bins=self.tokenizer_config.bird_y_bins,
            pipe_x_bins=self.tokenizer_config.pipe_x_bins,
            pipe_gap_bins=self.tokenizer_config.pipe_gap_bins,
            respawn_threshold_bins=self.tokenizer_config.respawn_threshold_bins,
            n_layer=0,
            n_head=0,
            n_embd=0,
            dropout=0.0,
        )

    def reset(self) -> None:
        self.frames = [FrameTokens(**frame.__dict__) for frame in self.initial_frames]
        if self.initial_present is None:
            self.pipe_present = [self.infer_present(frame) for frame in self.frames]
        else:
            self.pipe_present = list(self.initial_present)
        self.done = False
        self.last_action = None
        self.last_respawn = False
        self.trace_records: list[dict[str, Any]] = []
        self.cache: KVCache | None = None

    def context_tokens(self) -> tuple[list[int], list[int]]:
        tokens = ["<bos>"]
        for frame, present in zip(self.frames, self.pipe_present, strict=True):
            tokens.extend(self.frame_tokens_for_context(frame, present))
        ids = [self.vocab[token] for token in tokens]
        positions = list(range(len(ids)))
        block = self.engine.config.block_size
        if len(ids) > block:
            drop = len(ids) - block
            ids = ids[-block:]
            positions = positions[-block:]
            if self.cache is not None:
                self.cache = self.cache.truncate_left(drop)
        return ids, positions

    def infer_present(self, frame: FrameTokens) -> tuple[bool, bool]:
        def visible(x: int) -> bool:
            return 0 < x < self.tokenizer_config.pipe_x_bins - 1

        return visible(frame.pipe0_x), visible(frame.pipe1_x)

    def frame_tokens_for_context(self, frame: FrameTokens, present: tuple[bool, bool]) -> list[str]:
        tokens = [f"bird_y_{frame.bird_y:03d}"]
        for pipe_idx in range(2):
            x = frame.pipe0_x if pipe_idx == 0 else frame.pipe1_x
            gap = frame.pipe0_gap if pipe_idx == 0 else frame.pipe1_gap
            gap_token = f"pipe{pipe_idx}_gap_{gap:03d}"
            hidden = not present[pipe_idx]
            tokens.append(f"pipe{pipe_idx}_present_{0 if hidden else 1}")
            if hidden:
                tokens.append(f"pipe{pipe_idx}_x_hidden")
                tokens.append(f"pipe{pipe_idx}_gap_hidden")
            else:
                tokens.append(f"pipe{pipe_idx}_x_{x:03d}")
                tokens.append(gap_token)
        tokens.extend([f"respawn_{frame.respawn}", f"done_{frame.done}", f"action_{frame.action}"])
        return tokens

    def encode_context(self, ids: list[int], positions: list[int]) -> Any:
        logits, self.cache = self.engine.prefill(
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        )
        return FlatLMInferenceEngine.logits_at_last(logits)[0]

    def forward_token(self, token_id: int, position: int) -> Any:
        if self.cache is None:
            raise RuntimeError("decode step called before prefill")
        logits, self.cache = self.engine.step(
            torch.tensor([token_id], dtype=torch.long),
            torch.tensor([position], dtype=torch.long),
            self.cache,
        )
        return FlatLMInferenceEngine.logits_at_last(logits)[0]

    def append_token(self, ids: list[int], positions: list[int], token: str) -> Any:
        next_pos = positions[-1] + 1 if positions else 0
        token_id = self.vocab[token]
        ids.append(token_id)
        positions.append(next_pos)
        return self.forward_token(token_id, next_pos)

    def sample_token(self, ids: list[int], positions: list[int], logits: Any) -> tuple[str, Any]:
        token_id = sample_from_logits(logits)
        token = self.id_to_token[token_id]
        next_pos = positions[-1] + 1 if positions else 0
        ids.append(token_id)
        positions.append(next_pos)
        next_logits = self.forward_token(token_id, next_pos)
        return token, next_logits

    def step(self, flap: bool) -> StepResult:
        if self.done:
            return StepResult(action=self.last_action or "A_IDLE", done=True, respawn=self.last_respawn)
        action = 1 if flap else 0
        action_label = "A_FLAP" if action else "A_IDLE"
        self.frames[-1].action = action
        prev_present = self.pipe_present[-1]
        ids, positions = self.context_tokens()
        self.cache = None
        generated_tokens: list[str] = []
        logits = self.encode_context(ids, positions)
        bird_y_token, logits = self.sample_token(ids, positions, logits)
        bird_y = parse_value_token(bird_y_token)
        generated_tokens.append(bird_y_token)
        pipe0_present_token, logits = self.sample_token(ids, positions, logits)
        generated_tokens.append(pipe0_present_token)
        pipe0_present = parse_value_token(pipe0_present_token) == 1
        if pipe0_present:
            pipe0_x_token, logits = self.sample_token(ids, positions, logits)
            pipe0_x = parse_value_token(pipe0_x_token)
            generated_tokens.append(pipe0_x_token)
            pipe0_respawn = not prev_present[0] or (pipe0_x - self.frames[-1].pipe0_x) >= self.typed_config.respawn_threshold_bins
            pipe0_gap_token, logits = self.sample_token(ids, positions, logits)
            pipe0_gap = parse_value_token(pipe0_gap_token)
            generated_tokens.append(pipe0_gap_token)
        else:
            pipe0_x = self.tokenizer_config.pipe_x_bins - 1
            pipe0_gap = self.frames[-1].pipe0_gap
            pipe0_respawn = False
            logits = self.append_token(ids, positions, "pipe0_x_hidden")
            logits = self.append_token(ids, positions, "pipe0_gap_hidden")
            generated_tokens.extend(["pipe0_x_hidden", "pipe0_gap_hidden"])

        pipe1_present_token, logits = self.sample_token(ids, positions, logits)
        generated_tokens.append(pipe1_present_token)
        pipe1_present = parse_value_token(pipe1_present_token) == 1
        if pipe1_present:
            pipe1_x_token, logits = self.sample_token(ids, positions, logits)
            pipe1_x = parse_value_token(pipe1_x_token)
            generated_tokens.append(pipe1_x_token)
            pipe1_respawn = not prev_present[1] or (pipe1_x - self.frames[-1].pipe1_x) >= self.typed_config.respawn_threshold_bins
            pipe1_gap_token, logits = self.sample_token(ids, positions, logits)
            pipe1_gap = parse_value_token(pipe1_gap_token)
            generated_tokens.append(pipe1_gap_token)
        else:
            pipe1_x = self.tokenizer_config.pipe_x_bins - 1
            pipe1_gap = self.frames[-1].pipe1_gap
            pipe1_respawn = False
            logits = self.append_token(ids, positions, "pipe1_x_hidden")
            logits = self.append_token(ids, positions, "pipe1_gap_hidden")
            generated_tokens.extend(["pipe1_x_hidden", "pipe1_gap_hidden"])

        respawn_token, logits = self.sample_token(ids, positions, logits)
        done_token, _logits = self.sample_token(ids, positions, logits)
        generated_tokens.extend([respawn_token, done_token])
        self.append_token(ids, positions, f"action_{action}")
        generated_tokens.append(f"action_{action}")
        self.done = parse_value_token(done_token) == 1
        respawn = pipe0_respawn or pipe1_respawn or parse_value_token(respawn_token) == 1
        self.frames.append(
            FrameTokens(
                bird_y=bird_y,
                pipe0_x=pipe0_x,
                pipe0_gap=pipe0_gap,
                pipe1_x=pipe1_x,
                pipe1_gap=pipe1_gap,
                respawn=int(respawn),
                done=int(self.done),
                action=0,
            )
        )
        self.pipe_present.append((pipe0_present, pipe1_present))
        self.last_action = action_label
        self.last_respawn = respawn
        self.trace_records.append(
            {
                "step": len(self.frames) - 2,
                "input_action": action,
                "generated_tokens": generated_tokens,
                "pipe_present": {
                    "pipe0": bool(pipe0_present),
                    "pipe1": bool(pipe1_present),
                },
                "frame": {
                    "bird_y": bird_y,
                    "pipe0_x": pipe0_x,
                    "pipe0_gap": pipe0_gap,
                    "pipe1_x": pipe1_x,
                    "pipe1_gap": pipe1_gap,
                    "respawn": int(respawn),
                    "done": int(self.done),
                    "action": action,
                },
            }
        )
        return StepResult(action=action_label, done=self.done, respawn=respawn)

    def latest_state(self, pipe_gap_px: int) -> dict[str, int | str]:
        current = self.frames[-1]
        prev = self.frames[-2] if len(self.frames) > 1 else current
        state = frame_to_render_state(current, prev, self.typed_config, pipe_gap_px, self.last_action, self.done)
        present = self.pipe_present[-1]
        if not present[1]:
            state["p1_x"] = SCREEN_WIDTH
            state["p1_top"] = 0
            state["p1_bottom"] = SCREEN_HEIGHT
        if not present[0]:
            state["p0_x"] = SCREEN_WIDTH
            state["p0_top"] = 0
            state["p0_bottom"] = SCREEN_HEIGHT
        return state


def main() -> None:
    global torch
    args = parse_args()
    try:
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: torch. Install the project dependencies, then rerun python -m scripts.play.") from exc
    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pygame. Install pygame to run python -m scripts.play.") from exc

    torch = torch_module
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    model, tokenizer_config, vocab, id_to_token, position_scheme = load_checkpoint(Path(args.checkpoint), device)
    print(
        f"device={device} renderer=pygame player_version={PLAYER_VERSION} "
        f"position_scheme={position_scheme}",
        flush=True,
    )
    print(f"loading_prompt={args.prompt_from_data} line={args.line} state_index={args.state_index}", flush=True)
    seed_present = None
    visible_seed = visible_seed_frames_from_data(
        Path(args.prompt_from_data),
        args.line,
        args.state_index,
        max(1, model.config.block_size // 10),
        tokenizer_config,
    )
    if visible_seed is None:
        seed_frames = seed_frames_from_data(
            Path(args.prompt_from_data),
            tokenizer_config,
            args.line,
            args.state_index,
            max(1, model.config.block_size // 10),
        )
    else:
        seed_frames, seed_present = visible_seed
    print(f"seed_frames={len(seed_frames)}", flush=True)
    engine = FlatLMInferenceEngine(model, device)
    stepper = FlatLMStepper(
        engine=engine,
        vocab=vocab,
        id_to_token=id_to_token,
        tokenizer_config=tokenizer_config,
        seed_frames=seed_frames,
        seed_present=seed_present,
        device=device,
    )
    print("starting_pygame=yes", flush=True)
    pygame.init()
    try:
        PygamePlayer(pygame, stepper, args).run()
    finally:
        if args.trace_output:
            trace_path = Path(args.trace_output)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as f:
                for record in stepper.trace_records:
                    f.write(json.dumps(record) + "\n")
            print(f"trace_output={trace_path}", flush=True)


if __name__ == "__main__":
    main()
