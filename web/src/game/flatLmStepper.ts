import { OnnxFlatLMEngine } from '../model/onnxEngine.ts'
import type {
  FrameTokens,
  Manifest,
  RenderState,
  StepResult,
  StepTimings,
  TokenizerConfig,
  TraceRecord,
} from '../types/manifest.ts'
import { frameToRenderState, SCREEN_HEIGHT, SCREEN_WIDTH } from '../render/flappyCanvas.ts'

function parseValueToken(token: string): number {
  return Number.parseInt(token.split('_').at(-1) ?? '0', 10)
}

function sampleFromLogits(logits: Float32Array): number {
  let bestId = 0
  let bestLogit = Number.NEGATIVE_INFINITY
  for (let id = 0; id < logits.length; id += 1) {
    const logit = logits[id]
    if (logit > bestLogit) {
      bestLogit = logit
      bestId = id
    }
  }
  return bestId
}

export type FlatLMStepperOptions = {
  pipeGapPx?: number
}

export class FlatLMStepper {
  private readonly engine: OnnxFlatLMEngine
  private readonly vocab: Record<string, number>
  private readonly idToToken: Record<string, string>
  private readonly tokenizerConfig: TokenizerConfig
  private readonly initialFrames: FrameTokens[]
  private readonly initialPresent: [boolean, boolean][] | null
  private readonly pipeGapPx: number

  frames: FrameTokens[] = []
  pipePresent: [boolean, boolean][] = []
  done = false
  lastAction: string | null = null
  lastRespawn = false
  traceRecords: TraceRecord[] = []
  lastTimings: StepTimings | null = null
  private contextIds: number[] = []
  private contextPositions: number[] = []
  private nextPosition = 0

  constructor(manifest: Manifest, engine: OnnxFlatLMEngine, seedFrames: FrameTokens[], seedPresent: [boolean, boolean][] | null, options: FlatLMStepperOptions = {}) {
    this.engine = engine
    this.vocab = manifest.vocab
    this.idToToken = manifest.id_to_token
    this.tokenizerConfig = manifest.tokenizer_config
    this.initialFrames = seedFrames.map((frame) => ({ ...frame }))
    this.initialPresent = seedPresent ? seedPresent.map((item) => [...item] as [boolean, boolean]) : null
    this.pipeGapPx = options.pipeGapPx ?? 100
    this.reset()
  }

  reset(): void {
    this.frames = this.initialFrames.map((frame) => ({ ...frame }))
    this.pipePresent = this.initialPresent
      ? this.initialPresent.map((item) => [...item] as [boolean, boolean])
      : this.frames.map((frame) => this.inferPresent(frame))
    this.done = false
    this.lastAction = null
    this.lastRespawn = false
    this.traceRecords = []
    this.lastTimings = null
    this.engine.resetCache()
    this.rebuildContextState()
  }

  private get respawnThresholdBins(): number {
    return this.tokenizerConfig.respawn_threshold_bins
  }

  private contextTokens(includeLastAction: boolean): { ids: number[]; positions: number[]; nextPosition: number } {
    const tokens = ['<bos>']
    for (let i = 0; i < this.frames.length; i += 1) {
      tokens.push(...this.frameTokensForContext(
        this.frames[i],
        this.pipePresent[i],
        includeLastAction || i < this.frames.length - 1,
      ))
    }
    const ids = tokens.map((token) => requireTokenId(this.vocab, token))
    const positions = ids.map((_, index) => index)
    const nextPosition = positions.length
    const block = this.engine.blockSize
    if (ids.length > block) {
      return {
        ids: ids.slice(-block),
        positions: positions.slice(-block),
        nextPosition,
      }
    }
    return { ids, positions, nextPosition }
  }

  private rebuildContextState(): void {
    const { ids, positions, nextPosition } = this.contextTokens(false)
    this.contextIds = ids
    this.contextPositions = positions
    this.nextPosition = nextPosition
  }

  inferPresent(frame: FrameTokens): [boolean, boolean] {
    const visible = (x: number) => x > 0 && x < this.tokenizerConfig.pipe_x_bins - 1
    return [visible(frame.pipe0_x), visible(frame.pipe1_x)]
  }

  frameTokensForContext(frame: FrameTokens, present: [boolean, boolean], includeAction = true): string[] {
    const tokens = [`bird_y_${frame.bird_y.toString().padStart(3, '0')}`]
    for (let pipeIdx = 0; pipeIdx < 2; pipeIdx += 1) {
      const x = pipeIdx === 0 ? frame.pipe0_x : frame.pipe1_x
      const gap = pipeIdx === 0 ? frame.pipe0_gap : frame.pipe1_gap
      const gapToken = `pipe${pipeIdx}_gap_${gap.toString().padStart(3, '0')}`
      const hidden = !present[pipeIdx]
      tokens.push(`pipe${pipeIdx}_present_${hidden ? 0 : 1}`)
      if (hidden) {
        tokens.push(`pipe${pipeIdx}_x_hidden`, `pipe${pipeIdx}_gap_hidden`)
      } else {
        tokens.push(`pipe${pipeIdx}_x_${x.toString().padStart(3, '0')}`, gapToken)
      }
    }
    tokens.push(
      `respawn_${frame.respawn}`,
      `done_${frame.done}`,
    )
    if (includeAction) {
      tokens.push(`action_${frame.action}`)
    }
    return tokens
  }

  private trimContextWindow(): void {
    const block = this.engine.blockSize
    if (this.contextIds.length > block) {
      const drop = this.contextIds.length - block
      this.contextIds = this.contextIds.slice(drop)
      this.contextPositions = this.contextPositions.slice(drop)
    }
  }

  private async ensureContextPrefilled(): Promise<Float32Array> {
    if (this.engine.currentLogits) {
      this.engine.markCacheReused()
      return this.engine.currentLogits
    }
    if (this.contextIds.length === 0) {
      throw new Error('Cannot prefill empty context')
    }
    return this.engine.prefill(this.contextIds, this.contextPositions)
  }

  private async forwardToken(tokenId: number, position: number): Promise<Float32Array> {
    return this.engine.step(tokenId, position)
  }

  private async appendToken(token: string): Promise<Float32Array> {
    const tokenId = requireTokenId(this.vocab, token)
    const position = this.nextPosition
    this.contextIds.push(tokenId)
    this.contextPositions.push(position)
    this.nextPosition += 1
    this.trimContextWindow()
    return this.forwardToken(tokenId, position)
  }

  private async sampleToken(
    logits: Float32Array,
  ): Promise<[string, Float32Array]> {
    const tokenId = sampleFromLogits(logits)
    const token = this.idToToken[String(tokenId)]
    const position = this.nextPosition
    this.contextIds.push(tokenId)
    this.contextPositions.push(position)
    this.nextPosition += 1
    this.trimContextWindow()
    const nextLogits = await this.forwardToken(tokenId, position)
    return [token, nextLogits]
  }

  async step(flap: boolean): Promise<StepResult> {
    if (this.done) {
      return { action: this.lastAction ?? 'A_IDLE', done: true, respawn: this.lastRespawn }
    }
    const stepStart = performance.now()
    this.engine.resetStepTimings()
    const action = flap ? 1 : 0
    const actionLabel = flap ? 'A_FLAP' : 'A_IDLE'
    let logits = await this.ensureContextPrefilled()
    this.frames[this.frames.length - 1].action = action
    const prevPresent = this.pipePresent[this.pipePresent.length - 1]
    const generatedTokens: string[] = []
    generatedTokens.push(`action_${action}`)
    logits = await this.appendToken(`action_${action}`)

    const [birdYToken, birdLogits] = await this.sampleToken(logits)
    const birdY = parseValueToken(birdYToken)
    generatedTokens.push(birdYToken)
    logits = birdLogits

    const [pipe0PresentToken, pipe0PresentLogits] = await this.sampleToken(logits)
    generatedTokens.push(pipe0PresentToken)
    const pipe0Present = parseValueToken(pipe0PresentToken) === 1
    logits = pipe0PresentLogits

    let pipe0X: number
    let pipe0Gap: number
    let pipe0Respawn: boolean
    if (pipe0Present) {
      const [pipe0XToken, pipe0XLogits] = await this.sampleToken(logits)
      pipe0X = parseValueToken(pipe0XToken)
      generatedTokens.push(pipe0XToken)
      pipe0Respawn =
        !prevPresent[0] ||
        pipe0X - this.frames[this.frames.length - 1].pipe0_x >= this.respawnThresholdBins
      const [pipe0GapToken, pipe0GapLogits] = await this.sampleToken(pipe0XLogits)
      pipe0Gap = parseValueToken(pipe0GapToken)
      generatedTokens.push(pipe0GapToken)
      logits = pipe0GapLogits
    } else {
      pipe0X = this.tokenizerConfig.pipe_x_bins - 1
      pipe0Gap = this.frames[this.frames.length - 1].pipe0_gap
      pipe0Respawn = false
      logits = await this.appendToken('pipe0_x_hidden')
      logits = await this.appendToken('pipe0_gap_hidden')
      generatedTokens.push('pipe0_x_hidden', 'pipe0_gap_hidden')
    }

    let pipe1X: number
    let pipe1Gap: number
    let pipe1Respawn = false
    const [pipe1PresentToken, pipe1PresentLogits] = await this.sampleToken(logits)
    generatedTokens.push(pipe1PresentToken)
    const pipe1Present = parseValueToken(pipe1PresentToken) === 1
    logits = pipe1PresentLogits
    if (pipe1Present) {
      const [pipe1XToken, pipe1XLogits] = await this.sampleToken(logits)
      pipe1X = parseValueToken(pipe1XToken)
      generatedTokens.push(pipe1XToken)
      pipe1Respawn =
        !prevPresent[1] ||
        pipe1X - this.frames[this.frames.length - 1].pipe1_x >= this.respawnThresholdBins
      const [pipe1GapToken, pipe1GapLogits] = await this.sampleToken(pipe1XLogits)
      pipe1Gap = parseValueToken(pipe1GapToken)
      generatedTokens.push(pipe1GapToken)
      logits = pipe1GapLogits
    } else {
      pipe1X = this.tokenizerConfig.pipe_x_bins - 1
      pipe1Gap = this.frames[this.frames.length - 1].pipe1_gap
      pipe1Respawn = false
      logits = await this.appendToken('pipe1_x_hidden')
      logits = await this.appendToken('pipe1_gap_hidden')
      generatedTokens.push('pipe1_x_hidden', 'pipe1_gap_hidden')
    }

    const [respawnToken, respawnLogits] = await this.sampleToken(logits)
    const [doneToken] = await this.sampleToken(respawnLogits)
    generatedTokens.push(respawnToken, doneToken)

    this.done = parseValueToken(doneToken) === 1
    const respawn = pipe0Respawn || pipe1Respawn || parseValueToken(respawnToken) === 1
    this.frames.push({
      bird_y: birdY,
      pipe0_x: pipe0X,
      pipe0_gap: pipe0Gap,
      pipe1_x: pipe1X,
      pipe1_gap: pipe1Gap,
      respawn: respawn ? 1 : 0,
      done: this.done ? 1 : 0,
      action: 0,
    })
    this.pipePresent.push([pipe0Present, pipe1Present])
    this.lastAction = actionLabel
    this.lastRespawn = respawn
    const inference = this.engine.getStepTimings()
    const logicMs = performance.now() - stepStart - inference.prefillMs - inference.decodeMs
    this.lastTimings = {
      ...inference,
      logicMs: Math.max(0, logicMs),
      renderMs: 0,
      totalMs: performance.now() - stepStart,
    }
    this.traceRecords.push({
      step: this.frames.length - 2,
      input_action: action,
      generated_tokens: generatedTokens,
      pipe_present: {
        pipe0: pipe0Present,
        pipe1: pipe1Present,
      },
      frame: {
        bird_y: birdY,
        pipe0_x: pipe0X,
        pipe0_gap: pipe0Gap,
        pipe1_x: pipe1X,
        pipe1_gap: pipe1Gap,
        respawn: respawn ? 1 : 0,
        done: this.done ? 1 : 0,
        action,
      },
      timings: { ...this.lastTimings },
    })
    return { action: actionLabel, done: this.done, respawn }
  }

  latestState(): RenderState {
    const current = this.frames[this.frames.length - 1]
    const prev = this.frames.length > 1 ? this.frames[this.frames.length - 2] : current
    const state = frameToRenderState(
      current,
      prev,
      this.tokenizerConfig,
      this.pipeGapPx,
      this.lastAction,
      this.done,
    )
    const present = this.pipePresent[this.pipePresent.length - 1]
    if (!present[1]) {
      state.p1_x = SCREEN_WIDTH
      state.p1_top = 0
      state.p1_bottom = SCREEN_HEIGHT
    }
    if (!present[0]) {
      state.p0_x = SCREEN_WIDTH
      state.p0_top = 0
      state.p0_bottom = SCREEN_HEIGHT
    }
    return state
  }
}

function requireTokenId(vocab: Record<string, number>, token: string): number {
  const id = vocab[token]
  if (id === undefined) {
    throw new Error(`Unknown token "${token}"`)
  }
  return id
}
