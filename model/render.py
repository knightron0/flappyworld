"""Render a Flappy Bird model episode from JSONL.

Example:
  python -m scripts.render_episode dataset/flappy_lm_2pipes_ppo.jsonl --line 0 --output episode.gif
"""

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
GROUND_Y = int(SCREEN_HEIGHT * 0.79)
PIPE_WIDTH = 52
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BIRD_X = int(SCREEN_WIDTH * 0.2)

STATE_RE = re.compile(
    r"p0_x_(?P<p0_x>\d+) "
    r"p0_top_(?P<p0_top>\d+) "
    r"p0_bottom_(?P<p0_bottom>\d+) "
    r"p1_x_(?P<p1_x>\d+) "
    r"p1_top_(?P<p1_top>\d+) "
    r"p1_bottom_(?P<p1_bottom>\d+) "
    r"bird_y_(?P<bird_y>\d+) "
    r"bird_rot_(?P<bird_rot>[+-]\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=str)
    parser.add_argument("--line", type=int, default=0, help="0-indexed JSONL line to render")
    parser.add_argument("--output", type=str, default="episode.gif")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--save-frames-dir", type=str, default=None)
    parser.add_argument("--hide-guides", action="store_true")
    return parser.parse_args()


def load_episode(path: Path, line_index: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == line_index:
                return json.loads(line)
    raise IndexError(f"{path} has no line {line_index}")


def parse_states(tokens: list[str]) -> list[dict]:
    states = []
    idx = 0
    while idx + 7 < len(tokens):
        chunk = tokens[idx : idx + 8]
        text = " ".join(chunk)
        match = STATE_RE.fullmatch(text)
        if match:
            state = {key: int(value) for key, value in match.groupdict().items()}
            action = tokens[idx + 8] if idx + 8 < len(tokens) and tokens[idx + 8].startswith("A_") else None
            reward = tokens[idx + 9] if idx + 9 < len(tokens) and tokens[idx + 9].startswith("R_") else None
            state["action"] = action
            state["reward"] = reward
            states.append(state)
            idx += 10
        else:
            idx += 1
    return states


def draw_background(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), fill=(111, 197, 206))
    for cloud_x, cloud_y in ((20, 70), (150, 45), (225, 95)):
        draw.ellipse((cloud_x, cloud_y, cloud_x + 44, cloud_y + 24), fill=(235, 250, 250))
        draw.ellipse((cloud_x + 18, cloud_y - 8, cloud_x + 60, cloud_y + 22), fill=(235, 250, 250))
    draw.rectangle((0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT), fill=(222, 216, 149))
    draw.rectangle((0, GROUND_Y, SCREEN_WIDTH, GROUND_Y + 14), fill=(118, 189, 74))


def draw_pipe(
    draw: ImageDraw.ImageDraw,
    x: int,
    top: int,
    bottom: int,
    label: str,
    show_guides: bool,
) -> None:
    if top == 0 and bottom == SCREEN_HEIGHT:
        return

    left = x
    right = x + PIPE_WIDTH
    center = x + PIPE_WIDTH // 2
    if right < 0 or left > SCREEN_WIDTH:
        return

    pipe = (93, 201, 72)
    shade = (54, 141, 55)
    lip = (82, 185, 66)

    draw.rectangle((left, 0, right, top), fill=pipe, outline=shade)
    draw.rectangle((left - 4, max(0, top - 20), right + 4, top), fill=lip, outline=shade)
    draw.rectangle((left, bottom, right, GROUND_Y), fill=pipe, outline=shade)
    draw.rectangle((left - 4, bottom, right + 4, bottom + 20), fill=lip, outline=shade)

    if show_guides:
        # Token x is the left edge. Center is derived as x + PIPE_WIDTH / 2.
        draw.line((left, 0, left, GROUND_Y), fill=(210, 40, 40), width=1)
        draw.line((center, 0, center, GROUND_Y), fill=(40, 70, 210), width=1)
        if 0 <= left <= SCREEN_WIDTH:
            draw.text((left + 2, max(2, top - 34)), f"{label}_x", fill=(210, 40, 40), font=ImageFont.load_default())
        if 0 <= center <= SCREEN_WIDTH:
            draw.text((center + 2, max(14, top - 22)), f"{label}_center", fill=(40, 70, 210), font=ImageFont.load_default())


def rotated_rect(cx: float, cy: float, width: float, height: float, degrees: float) -> list[tuple[float, float]]:
    radians = math.radians(-degrees)
    cos_r = math.cos(radians)
    sin_r = math.sin(radians)
    points = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
        points.append((cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r))
    return points


def draw_bird(draw: ImageDraw.ImageDraw, y: int, rot: int) -> None:
    cx = BIRD_X + BIRD_WIDTH / 2
    cy = y + BIRD_HEIGHT / 2
    body = rotated_rect(cx, cy, BIRD_WIDTH, BIRD_HEIGHT, rot)
    wing = rotated_rect(cx - 3, cy + 2, 16, 9, rot)
    beak = rotated_rect(cx + 18, cy, 10, 7, rot)
    draw.polygon(body, fill=(250, 220, 65), outline=(115, 89, 31))
    draw.polygon(wing, fill=(244, 170, 54), outline=(115, 89, 31))
    draw.polygon(beak, fill=(236, 108, 46), outline=(115, 60, 30))
    draw.ellipse((cx + 4, cy - 8, cx + 11, cy - 1), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.ellipse((cx + 8, cy - 5, cx + 10, cy - 3), fill=(0, 0, 0))


def draw_frame(state: dict, frame_idx: int, scale: int, show_guides: bool) -> Image.Image:
    image = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT))
    draw = ImageDraw.Draw(image)
    draw_background(draw)
    draw_pipe(draw, state["p0_x"], state["p0_top"], state["p0_bottom"], "p0", show_guides)
    draw_pipe(draw, state["p1_x"], state["p1_top"], state["p1_bottom"], "p1", show_guides)
    draw_bird(draw, state["bird_y"], state["bird_rot"])

    label = f"t={frame_idx}"
    if state.get("action"):
        label += f" {state['action']}"
    if state.get("reward"):
        label += f" {state['reward']}"
    draw.text((6, 6), label, fill=(20, 45, 55), font=ImageFont.load_default())

    if scale != 1:
        image = image.resize((SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale), Image.Resampling.NEAREST)
    return image


def save_output(frames: list[Image.Image], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=False,
        )
    else:
        frames[0].save(output)


def main() -> None:
    args = parse_args()
    episode = load_episode(Path(args.jsonl), args.line)
    states = parse_states(episode["tokens"])[: args.max_frames]
    if not states:
        raise ValueError("No state tokens found in selected episode")

    frames = [
        draw_frame(state, idx, args.scale, not args.hide_guides)
        for idx, state in enumerate(states)
    ]

    if args.save_frames_dir:
        frames_dir = Path(args.save_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{idx:04d}.png")

    save_output(frames, Path(args.output), args.fps)
    print(f"rendered_frames={len(frames)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
