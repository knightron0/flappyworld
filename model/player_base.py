"""Shared rendering helpers for the flat autoregressive Flappy LM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model.render import GROUND_Y, SCREEN_HEIGHT, SCREEN_WIDTH, draw_frame
from model.shared import FrameTokens, TokenizerConfig, load_episodes


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
    tokenizer_config: TokenizerConfig,
    pipe_gap_px: int,
    action_label: str | None,
    done: bool,
) -> dict[str, int | str]:
    def gap_edges(gap_bin: int) -> tuple[int, int]:
        center = bin_to_px(gap_bin, tokenizer_config.pipe_gap_bins, SCREEN_HEIGHT)
        top = max(0, min(GROUND_Y, center - pipe_gap_px // 2))
        bottom = max(0, min(GROUND_Y, center + pipe_gap_px // 2))
        return top, bottom

    p0_top, p0_bottom = gap_edges(frame.pipe0_gap)
    p1_top, p1_bottom = gap_edges(frame.pipe1_gap)
    bird_y = bin_to_px(frame.bird_y, tokenizer_config.bird_y_bins, SCREEN_HEIGHT)
    prev_bird_y = bin_to_px(prev_frame.bird_y, tokenizer_config.bird_y_bins, SCREEN_HEIGHT)
    return {
        "p0_x": bin_to_px(frame.pipe0_x, tokenizer_config.pipe_x_bins, SCREEN_WIDTH),
        "p0_top": p0_top,
        "p0_bottom": p0_bottom,
        "p1_x": bin_to_px(frame.pipe1_x, tokenizer_config.pipe_x_bins, SCREEN_WIDTH),
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


class PygamePlayer:
    def __init__(self, pygame: Any, stepper: Any, args: Any):
        self.pygame = pygame
        self.stepper = stepper
        self.args = args
        self.pending_flap = False
        self.paused = False
        self.frame_idx = 0
        self.screen = pygame.display.set_mode((SCREEN_WIDTH * args.scale, SCREEN_HEIGHT * args.scale))
        pygame.display.set_caption("Flat AR World Player")
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
