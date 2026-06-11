"""Interactively play the autoregressive Flappy world model.

Controls:
  Space / Up / mouse click: queue a flap for the next model step
  P: pause/resume
  S: sample one frame while paused
  R: reset to the prompt state
  Esc / Q: quit

The selected action is written into the latest frame before predicting the next
frame. This matches training: a frame's action is the action used for the
transition out of that frame.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model.data import (
    FrameTokens,
    ModelConfig,
    TokenizedARWorldModel,
    TokenizerConfig,
    frame_to_tensor,
    load_episodes,
)
from model.render import GROUND_Y, SCREEN_HEIGHT, SCREEN_WIDTH, draw_frame


torch: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model/best.pt")
    parser.add_argument("--prompt-from-data", type=str, default="dataset/flappy_lm_2pipes_ppo.jsonl")
    parser.add_argument("--line", type=int, default=0)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--renderer", choices=("pygame", "matplotlib"), default="pygame")
    parser.add_argument("--backend", type=str, default=None, help="Matplotlib GUI backend, e.g. TkAgg or QtAgg.")
    parser.add_argument("--pipe-gap-px", type=int, default=100)
    parser.add_argument("--done-threshold", type=float, default=0.5)
    parser.add_argument("--sample-numeric", action="store_true", help="Sample bird_y and pipe_x instead of using argmax.")
    parser.add_argument("--greedy-gaps", action="store_true", help="Use argmax instead of sampling newly respawned pipe gaps.")
    parser.add_argument("--hide-guides", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def default_device() -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: Any) -> tuple[TokenizedARWorldModel, TokenizerConfig]:
    ckpt = torch.load(path, map_location=device)
    model_config = ModelConfig(**ckpt["model_config"])
    tokenizer_config = TokenizerConfig(**ckpt["tokenizer_config"])
    model = TokenizedARWorldModel(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer_config


def sample_categorical(logits: Any, temperature: float, greedy: bool) -> int:
    if greedy:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def bin_to_px(bin_idx: int, bins: int, scale: int) -> int:
    return int(round(float(bin_idx) * float(scale) / float(max(1, bins - 1))))


def infer_bird_rot(prev_y: int, bird_y: int) -> int:
    dy = bird_y - prev_y
    if dy < -1:
        return 45
    return max(-90, min(45, int(round(-6 * dy))))


def frame_to_render_state(
    frame: FrameTokens,
    prev_frame: FrameTokens,
    cfg: ModelConfig,
    pipe_gap_px: int,
    action_label: str | None,
    done: bool,
) -> dict[str, int | str]:
    def gap_edges(gap_bin: int) -> tuple[int, int]:
        center = bin_to_px(gap_bin, cfg.pipe_gap_bins, SCREEN_HEIGHT)
        top = max(0, min(GROUND_Y, center - pipe_gap_px // 2))
        bottom = max(0, min(GROUND_Y, center + pipe_gap_px // 2))
        return top, bottom

    p0_top, p0_bottom = gap_edges(frame.pipe0_gap)
    p1_top, p1_bottom = gap_edges(frame.pipe1_gap)
    bird_y = bin_to_px(frame.bird_y, cfg.bird_y_bins, SCREEN_HEIGHT)
    prev_bird_y = bin_to_px(prev_frame.bird_y, cfg.bird_y_bins, SCREEN_HEIGHT)
    return {
        "p0_x": bin_to_px(frame.pipe0_x, cfg.pipe_x_bins, SCREEN_WIDTH),
        "p0_top": p0_top,
        "p0_bottom": p0_bottom,
        "p1_x": bin_to_px(frame.pipe1_x, cfg.pipe_x_bins, SCREEN_WIDTH),
        "p1_top": p1_top,
        "p1_bottom": p1_bottom,
        "bird_y": bird_y,
        "bird_rot": infer_bird_rot(prev_bird_y, bird_y),
        "action": action_label,
        "reward": "R_DEAD" if done else "R_ALIVE",
    }


@dataclass
class StepResult:
    action: str
    done: bool
    respawn: bool


class TokenizedARStepper:
    def __init__(
        self,
        model: TokenizedARWorldModel,
        seed_frames: list[FrameTokens],
        device: Any,
        temperature: float,
        sample_numeric: bool,
        greedy_gaps: bool,
        done_threshold: float,
    ):
        if not seed_frames:
            raise ValueError("Need at least one seed frame")
        self.model = model
        self.device = device
        self.temperature = temperature
        self.sample_numeric = sample_numeric
        self.greedy_gaps = greedy_gaps
        self.done_threshold = done_threshold
        self.initial_frames = [FrameTokens(**frame.__dict__) for frame in seed_frames]
        self.reset()

    def reset(self) -> None:
        self.frames = [FrameTokens(**frame.__dict__) for frame in self.initial_frames]
        self.done = False
        self.last_action = None
        self.last_respawn = False

    def context_tensor(self) -> Any:
        frames = self.frames[-self.model.config.history_size :]
        context = [frame_to_tensor(frame) for frame in frames]
        if len(context) < self.model.config.history_size:
            context = [context[0]] * (self.model.config.history_size - len(context)) + context
        return torch.stack(context, dim=0).unsqueeze(0).to(self.device)

    def step(self, flap: bool) -> StepResult:
        if self.done:
            return StepResult(action=self.last_action or "A_IDLE", done=True, respawn=self.last_respawn)
        action = 1 if flap else 0
        action_label = "A_FLAP" if action else "A_IDLE"
        self.frames[-1].action = action
        with torch.inference_mode():
            logits = self.model(self.context_tensor())

        prev = self.frames[-1]
        greedy_numeric = not self.sample_numeric
        bird_y = sample_categorical(logits["bird_y"][0], self.temperature, greedy_numeric)
        pipe0_x = sample_categorical(logits["pipe0_x"][0], self.temperature, greedy_numeric)
        pipe1_x = sample_categorical(logits["pipe1_x"][0], self.temperature, greedy_numeric)
        pipe0_respawn = (pipe0_x - prev.pipe0_x) >= self.model.config.respawn_threshold_bins
        pipe1_respawn = (pipe1_x - prev.pipe1_x) >= self.model.config.respawn_threshold_bins
        respawn = pipe0_respawn or pipe1_respawn or bool((logits["respawn"][0] > 0).item())
        pipe0_gap = (
            sample_categorical(logits["pipe0_gap"][0], self.temperature, self.greedy_gaps)
            if pipe0_respawn
            else prev.pipe0_gap
        )
        pipe1_gap = (
            sample_categorical(logits["pipe1_gap"][0], self.temperature, self.greedy_gaps)
            if pipe1_respawn
            else prev.pipe1_gap
        )
        self.done = bool(torch.sigmoid(logits["done"][0]).item() >= self.done_threshold)
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
        self.last_action = action_label
        self.last_respawn = respawn
        return StepResult(action=action_label, done=self.done, respawn=respawn)

    def latest_state(self, pipe_gap_px: int) -> dict[str, int | str]:
        current = self.frames[-1]
        prev = self.frames[-2] if len(self.frames) > 1 else current
        return frame_to_render_state(
            current,
            prev,
            self.model.config,
            pipe_gap_px,
            self.last_action,
            self.done,
        )


class PlayerWindow:
    def __init__(self, plt: Any, stepper: TokenizedARStepper, args: argparse.Namespace):
        self.plt = plt
        self.stepper = stepper
        self.args = args
        self.pending_flap = False
        self.paused = False
        self.frame_idx = 0
        self.fig, self.ax = plt.subplots(num="Tokenized AR World Player")
        self.ax.axis("off")
        self.artist = self.ax.imshow(self.draw())
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        interval_ms = max(1, int(1000 / max(args.fps, 0.001)))
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.on_tick)
        self.timer.start()
        self.render("ready")

    def draw(self):
        return draw_frame(self.stepper.latest_state(self.args.pipe_gap_px), self.frame_idx, self.args.scale, not self.args.hide_guides)

    def render(self, status: str) -> None:
        self.artist.set_data(self.draw())
        queued = "queued_flap=yes" if self.pending_flap else "queued_flap=no"
        paused = "paused=yes" if self.paused else "paused=no"
        state = self.stepper.latest_state(self.args.pipe_gap_px)
        self.ax.set_title(
            f"tokenized AR | {status} | {queued} | {paused} | "
            f"bird_y={state['bird_y']} p0_x={state['p0_x']} p1_x={state['p1_x']}",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def queue_flap(self) -> None:
        self.pending_flap = True
        self.render("flap queued")

    def sample_one(self) -> None:
        if self.stepper.done:
            self.render("done; press R to reset")
            return
        flap = self.pending_flap
        self.pending_flap = False
        result = self.stepper.step(flap)
        self.frame_idx += 1
        status = f"{result.action}"
        if result.respawn:
            status += " respawn"
        if result.done:
            status += " done"
        self.render(status)

    def on_tick(self) -> None:
        if not self.paused:
            self.sample_one()

    def reset(self) -> None:
        self.stepper.reset()
        self.pending_flap = False
        self.frame_idx = 0
        self.render("reset")

    def on_click(self, event: Any) -> None:
        self.queue_flap()

    def on_key(self, event: Any) -> None:
        key = event.key
        if key in (" ", "space", "up"):
            self.queue_flap()
        elif key == "p":
            self.paused = not self.paused
            self.render("paused" if self.paused else "resumed")
        elif key == "s":
            self.sample_one()
        elif key == "r":
            self.reset()
        elif key in ("escape", "q"):
            self.plt.close(self.fig)


class PygamePlayer:
    def __init__(self, pygame: Any, stepper: TokenizedARStepper, args: argparse.Namespace):
        self.pygame = pygame
        self.stepper = stepper
        self.args = args
        self.pending_flap = False
        self.paused = False
        self.frame_idx = 0
        self.screen = pygame.display.set_mode((SCREEN_WIDTH * args.scale, SCREEN_HEIGHT * args.scale))
        pygame.display.set_caption("Tokenized AR World Player")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)

    def queue_flap(self) -> None:
        self.pending_flap = True

    def sample_one(self) -> str:
        if self.stepper.done:
            return "done; press R"
        flap = self.pending_flap
        self.pending_flap = False
        result = self.stepper.step(flap)
        self.frame_idx += 1
        status = result.action
        if result.respawn:
            status += " respawn"
        if result.done:
            status += " done"
        return status

    def reset(self) -> None:
        self.stepper.reset()
        self.pending_flap = False
        self.frame_idx = 0

    def draw(self, status: str) -> None:
        image = draw_frame(
            self.stepper.latest_state(self.args.pipe_gap_px),
            self.frame_idx,
            self.args.scale,
            not self.args.hide_guides,
        ).convert("RGB")
        surface = self.pygame.image.fromstring(image.tobytes(), image.size, "RGB")
        self.screen.blit(surface, (0, 0))
        overlay = (
            f"{status} | queued_flap={'yes' if self.pending_flap else 'no'} | "
            f"paused={'yes' if self.paused else 'no'} | device={self.args.device or 'auto'}"
        )
        text = self.font.render(overlay, True, (20, 45, 55))
        self.screen.blit(text, (8, 28))
        self.pygame.display.flip()

    def run(self) -> None:
        status = "ready"
        running = True
        while running:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    running = False
                elif event.type == self.pygame.MOUSEBUTTONDOWN:
                    self.queue_flap()
                    status = "flap queued"
                elif event.type == self.pygame.KEYDOWN:
                    if event.key in (self.pygame.K_SPACE, self.pygame.K_UP):
                        self.queue_flap()
                        status = "flap queued"
                    elif event.key == self.pygame.K_p:
                        self.paused = not self.paused
                        status = "paused" if self.paused else "resumed"
                    elif event.key == self.pygame.K_s:
                        status = self.sample_one()
                    elif event.key == self.pygame.K_r:
                        self.reset()
                        status = "reset"
                    elif event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                        running = False
            if not self.paused:
                status = self.sample_one()
            self.draw(status)
            self.clock.tick(max(1, int(self.args.fps)))
        self.pygame.quit()


def seed_frames_from_data(path: Path, tokenizer_config: TokenizerConfig, line: int, state_index: int, history_size: int) -> list[FrameTokens]:
    episodes = load_episodes(path, tokenizer_config, max_episodes=line + 1)
    if line >= len(episodes):
        raise IndexError(f"{path} has no usable episode at line {line}")
    frames = episodes[line].frames
    if state_index >= len(frames):
        raise IndexError(f"--state-index {state_index} out of range; episode has {len(frames)} frames")
    start = max(0, state_index - history_size + 1)
    return frames[start : state_index + 1]


def main() -> None:
    global torch
    args = parse_args()
    try:
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: torch. Install the project dependencies, then rerun python -m scripts.play.") from exc

    torch = torch_module
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    print(f"device={device} renderer={args.renderer}", flush=True)
    model, tokenizer_config = load_checkpoint(Path(args.checkpoint), device)
    seed_frames = seed_frames_from_data(
        Path(args.prompt_from_data),
        tokenizer_config,
        args.line,
        args.state_index,
        model.config.history_size,
    )
    stepper = TokenizedARStepper(
        model=model,
        seed_frames=seed_frames,
        device=device,
        temperature=args.temperature,
        sample_numeric=args.sample_numeric,
        greedy_gaps=args.greedy_gaps,
        done_threshold=args.done_threshold,
    )
    if args.renderer == "pygame":
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: pygame. Install pygame or run with `--renderer matplotlib`.") from exc
        pygame.init()
        PygamePlayer(pygame, stepper, args).run()
    else:
        try:
            os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="flappyworld-mpl-"))
            import matplotlib

            if args.backend is not None:
                matplotlib.use(args.backend, force=True)
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: matplotlib. Install matplotlib, then rerun python -m scripts.play.") from exc

        backend = plt.get_backend()
        print(f"matplotlib_backend={backend}", flush=True)
        if "agg" in backend.lower():
            raise SystemExit(
                "Matplotlib is using a non-GUI backend, so no window can pop up. "
                "Try `--backend TkAgg` from an Ubuntu desktop terminal, or install a GUI backend such as python3-tk."
            )
        PlayerWindow(plt, stepper, args)
        plt.show()


if __name__ == "__main__":
    main()
