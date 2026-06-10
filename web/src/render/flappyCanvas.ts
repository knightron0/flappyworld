import type { FrameTokens, RenderState, TokenizerConfig } from '../types/manifest.ts'

export const SCREEN_WIDTH = 288
export const SCREEN_HEIGHT = 512
export const GROUND_Y = Math.floor(SCREEN_HEIGHT * 0.79)
export const PIPE_WIDTH = 52
export const BIRD_WIDTH = 34
export const BIRD_HEIGHT = 24
export const BIRD_X = Math.floor(SCREEN_WIDTH * 0.2)

const SPRITE_BASE = '/flappy-assets/sprites'

type SpriteName =
  | 'backgroundDay'
  | 'base'
  | 'pipe'
  | 'gameover'
  | 'birdUp'
  | 'birdMid'
  | 'birdDown'

const spriteFiles: Record<SpriteName, string> = {
  backgroundDay: 'background-day.png',
  base: 'base.png',
  pipe: 'pipe-green.png',
  gameover: 'gameover.png',
  birdUp: 'yellowbird-upflap.png',
  birdMid: 'yellowbird-midflap.png',
  birdDown: 'yellowbird-downflap.png',
}

const sprites = new Map<SpriteName, HTMLImageElement>()

function sprite(name: SpriteName): HTMLImageElement | null {
  if (typeof Image === 'undefined') {
    return null
  }
  const cached = sprites.get(name)
  if (cached) {
    return cached
  }
  const image = new Image()
  image.src = `${SPRITE_BASE}/${spriteFiles[name]}`
  sprites.set(name, image)
  return image
}

function isReady(image: HTMLImageElement | null): image is HTMLImageElement {
  return Boolean(image?.complete && image.naturalWidth > 0)
}

function binToPx(binIdx: number, bins: number, scale: number): number {
  return Math.round((binIdx * scale) / Math.max(1, bins - 1))
}

export function frameToRenderState(
  frame: FrameTokens,
  _prevFrame: FrameTokens,
  cfg: TokenizerConfig,
  pipeGapPx: number,
  actionLabel: string | null,
  done: boolean,
): RenderState {
  const gapEdges = (gapBin: number): [number, number] => {
    const center = binToPx(gapBin, cfg.pipe_gap_bins, SCREEN_HEIGHT)
    const top = Math.max(0, Math.min(GROUND_Y, center - Math.floor(pipeGapPx / 2)))
    const bottom = Math.max(0, Math.min(GROUND_Y, center + Math.floor(pipeGapPx / 2)))
    return [top, bottom]
  }
  const [p0Top, p0Bottom] = gapEdges(frame.pipe0_gap)
  const [p1Top, p1Bottom] = gapEdges(frame.pipe1_gap)
  const birdY = binToPx(frame.bird_y, cfg.bird_y_bins, SCREEN_HEIGHT)
  return {
    p0_x: binToPx(frame.pipe0_x, cfg.pipe_x_bins, SCREEN_WIDTH),
    p0_top: p0Top,
    p0_bottom: p0Bottom,
    p1_x: binToPx(frame.pipe1_x, cfg.pipe_x_bins, SCREEN_WIDTH),
    p1_top: p1Top,
    p1_bottom: p1Bottom,
    bird_y: birdY,
    action: actionLabel,
    reward: done ? 'R_DEAD' : 'R_ALIVE',
  }
}

function drawBackground(ctx: CanvasRenderingContext2D): void {
  const backgroundDay = sprite('backgroundDay')
  if (isReady(backgroundDay)) {
    ctx.drawImage(backgroundDay, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    return
  }
  ctx.fillStyle = '#6fc5ce'
  ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
  const clouds: [number, number][] = [
    [20, 70],
    [150, 45],
    [225, 95],
  ]
  ctx.fillStyle = '#ebfafa'
  for (const [cloudX, cloudY] of clouds) {
    ctx.beginPath()
    ctx.ellipse(cloudX + 22, cloudY + 12, 22, 12, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(cloudX + 39, cloudY + 7, 21, 15, 0, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.fillStyle = '#ded895'
  ctx.fillRect(0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
  ctx.fillStyle = '#76bd4a'
  ctx.fillRect(0, GROUND_Y, SCREEN_WIDTH, 14)
}

function drawBase(ctx: CanvasRenderingContext2D, frameIdx: number): void {
  const image = sprite('base')
  if (!isReady(image)) {
    return
  }
  const offset = frameIdx % 48
  for (let x = -offset; x < SCREEN_WIDTH; x += image.naturalWidth) {
    ctx.drawImage(image, x, GROUND_Y)
  }
}

function drawPipe(
  ctx: CanvasRenderingContext2D,
  x: number,
  top: number,
  bottom: number,
  label: string,
  showGuides: boolean,
): void {
  if (top === 0 && bottom === SCREEN_HEIGHT) {
    return
  }
  const left = x
  const right = x + PIPE_WIDTH
  const center = x + Math.floor(PIPE_WIDTH / 2)
  if (right < 0 || left > SCREEN_WIDTH) {
    return
  }
  const pipe = sprite('pipe')
  if (isReady(pipe)) {
    ctx.save()
    ctx.translate(left, top)
    ctx.scale(1, -1)
    ctx.drawImage(pipe, 0, 0, PIPE_WIDTH, pipe.naturalHeight)
    ctx.restore()
    ctx.drawImage(pipe, left, bottom, PIPE_WIDTH, pipe.naturalHeight)
  } else {
    drawFallbackPipe(ctx, left, top, bottom)
  }
  if (showGuides) {
    drawPipeGuides(ctx, left, top, center, label)
  }
}

function drawFallbackPipe(
  ctx: CanvasRenderingContext2D,
  left: number,
  top: number,
  bottom: number,
): void {
  const pipe = '#5dc948'
  const shade = '#368d37'
  const lip = '#52b942'
  ctx.fillStyle = pipe
  ctx.strokeStyle = shade
  ctx.fillRect(left, 0, PIPE_WIDTH, top)
  ctx.strokeRect(left, 0, PIPE_WIDTH, top)
  ctx.fillStyle = lip
  ctx.fillRect(left - 4, Math.max(0, top - 20), PIPE_WIDTH + 8, 20)
  ctx.fillRect(left, bottom, PIPE_WIDTH, GROUND_Y - bottom)
  ctx.strokeRect(left, bottom, PIPE_WIDTH, GROUND_Y - bottom)
  ctx.fillRect(left - 4, bottom, PIPE_WIDTH + 8, 20)
}

function drawPipeGuides(
  ctx: CanvasRenderingContext2D,
  left: number,
  top: number,
  center: number,
  label: string,
): void {
  ctx.strokeStyle = '#d22828'
  ctx.beginPath()
  ctx.moveTo(left, 0)
  ctx.lineTo(left, GROUND_Y)
  ctx.stroke()
  ctx.strokeStyle = '#2846d2'
  ctx.beginPath()
  ctx.moveTo(center, 0)
  ctx.lineTo(center, GROUND_Y)
  ctx.stroke()
  ctx.fillStyle = '#d22828'
  ctx.font = '10px monospace'
  ctx.fillText(`${label}_x`, left + 2, Math.max(12, top - 8))
  ctx.fillStyle = '#2846d2'
  ctx.fillText(`${label}_center`, center + 2, Math.max(20, top + 4))
}

function drawPolygon(ctx: CanvasRenderingContext2D, points: [number, number][], fill: string, stroke: string): void {
  ctx.beginPath()
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1])
  }
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.strokeStyle = stroke
  ctx.fill()
  ctx.stroke()
}

function rectPoints(x: number, y: number, width: number, height: number): [number, number][] {
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height],
  ]
}

function drawBird(ctx: CanvasRenderingContext2D, y: number, frameIdx: number): void {
  const birdFrames = [
    sprite('birdUp'),
    sprite('birdMid'),
    sprite('birdDown'),
    sprite('birdMid'),
  ]
  const image = birdFrames[Math.floor(frameIdx / 4) % birdFrames.length]
  if (isReady(image)) {
    ctx.drawImage(image, BIRD_X, y, BIRD_WIDTH, BIRD_HEIGHT)
    return
  }
  const cx = BIRD_X + BIRD_WIDTH / 2
  const cy = y + BIRD_HEIGHT / 2
  drawPolygon(ctx, rectPoints(BIRD_X, y, BIRD_WIDTH, BIRD_HEIGHT), '#fadc41', '#73591f')
  drawPolygon(ctx, rectPoints(cx - 11, cy - 2.5, 16, 9), '#f4aa36', '#73591f')
  drawPolygon(ctx, rectPoints(cx + 13, cy - 3.5, 10, 7), '#ec6c2e', '#733c1e')
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#1e1e1e'
  ctx.beginPath()
  ctx.ellipse(cx + 7.5, cy - 4.5, 3.5, 3.5, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = '#000000'
  ctx.beginPath()
  ctx.ellipse(cx + 9, cy - 4, 1, 1, 0, 0, Math.PI * 2)
  ctx.fill()
}

function drawDeathTint(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = 'rgba(139, 28, 42, 0.28)'
  ctx.fillRect(0, 0, SCREEN_WIDTH, GROUND_Y)
  ctx.fillStyle = 'rgba(255, 250, 240, 0.26)'
  ctx.fillRect(0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
  const gameover = sprite('gameover')
  if (isReady(gameover)) {
    const x = Math.floor((SCREEN_WIDTH - gameover.naturalWidth) / 2)
    const y = Math.floor(SCREEN_HEIGHT / 2 - gameover.naturalHeight)
    ctx.drawImage(gameover, x, y)
  }
  ctx.fillStyle = '#fffaf0'
  ctx.strokeStyle = '#111'
  ctx.lineWidth = 1
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.font = '12px monospace'
  ctx.strokeText('Press R to reset', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 24)
  ctx.fillText('Press R to reset', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 24)
  ctx.textAlign = 'start'
  ctx.textBaseline = 'alphabetic'
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  state: RenderState,
  _frameIdx: number,
  scale: number,
  showGuides: boolean,
): void {
  const width = SCREEN_WIDTH * scale
  const height = SCREEN_HEIGHT * scale
  if (ctx.canvas.width !== width || ctx.canvas.height !== height) {
    ctx.canvas.width = width
    ctx.canvas.height = height
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, width, height)
  ctx.setTransform(scale, 0, 0, scale, 0, 0)
  drawBackground(ctx)
  drawPipe(ctx, state.p0_x, state.p0_top, state.p0_bottom, 'p0', showGuides)
  drawPipe(ctx, state.p1_x, state.p1_top, state.p1_bottom, 'p1', showGuides)
  drawBase(ctx, _frameIdx)
  drawBird(ctx, state.bird_y, _frameIdx)
  if (state.reward === 'R_DEAD') {
    drawDeathTint(ctx)
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)
}
