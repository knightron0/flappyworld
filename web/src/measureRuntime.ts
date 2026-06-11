import { OnnxFlatLMEngine, type ProviderMode } from './model/onnxEngine.ts'
import type { FrameTokens, Manifest } from './types/manifest.ts'

const MODEL_BASE_URL = `${import.meta.env.BASE_URL}model`
let manifestPromise: Promise<Manifest> | null = null

export type CacheMode = 'reuse' | 'recompute'

export type ProviderMeasurement = {
  provider: ProviderMode
  mode: CacheMode
  providersLabel: string
  promptTokens: number
  prefillMs: number
  decodeCount: number
  totalMs: number
  totalDecodeMs: number
  averageDecodeMs: number
  tokens: string[]
  decodeSteps: Array<{ step: number; token: string; ms: number }>
}

function sampleFromLogits(logits: Float32Array): number {
  let bestId = 0
  let bestLogit = Number.NEGATIVE_INFINITY
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > bestLogit) {
      bestLogit = logits[i]
      bestId = i
    }
  }
  return bestId
}

async function loadManifest(): Promise<Manifest> {
  if (!manifestPromise) {
    manifestPromise = (async () => {
      const response = await fetch(`${MODEL_BASE_URL}/manifest.json?cache=${Date.now()}`, { cache: 'no-store' })
      if (!response.ok) {
        throw new Error(`Failed to load manifest (${response.status})`)
      }
      return response.json() as Promise<Manifest>
    })()
  }
  return manifestPromise
}

function frameTokens(frame: FrameTokens, present: [boolean, boolean], includeAction: boolean): string[] {
  const tokens = [`bird_y_${frame.bird_y.toString().padStart(3, '0')}`]
  for (let pipeIdx = 0; pipeIdx < 2; pipeIdx += 1) {
    const x = pipeIdx === 0 ? frame.pipe0_x : frame.pipe1_x
    const gap = pipeIdx === 0 ? frame.pipe0_gap : frame.pipe1_gap
    const hidden = !present[pipeIdx]
    tokens.push(`pipe${pipeIdx}_present_${hidden ? 0 : 1}`)
    if (hidden) {
      tokens.push(`pipe${pipeIdx}_x_hidden`, `pipe${pipeIdx}_gap_hidden`)
    } else {
      tokens.push(`pipe${pipeIdx}_x_${x.toString().padStart(3, '0')}`, `pipe${pipeIdx}_gap_${gap.toString().padStart(3, '0')}`)
    }
  }
  tokens.push(`respawn_${frame.respawn}`, `done_${frame.done}`)
  if (includeAction) {
    tokens.push(`action_${frame.action}`)
  }
  return tokens
}

function buildSeedContext(manifest: Manifest): { ids: number[]; positions: number[]; nextPosition: number } {
  const seed = manifest.default_seed
  if (!seed) {
    throw new Error('manifest missing default_seed')
  }
  const tokens = ['<bos>']
  for (let i = 0; i < seed.frames.length; i += 1) {
    const present = seed.pipe_present?.[i] ?? [true, true]
    tokens.push(...frameTokens(seed.frames[i], present, i < seed.frames.length - 1))
  }
  const ids = tokens.map((token) => {
    const id = manifest.vocab[token]
    if (id === undefined) {
      throw new Error(`Missing token in vocab: ${token}`)
    }
    return id
  })
  const positions = ids.map((_, index) => index)
  return { ids, positions, nextPosition: positions.length }
}

function trimContext(ids: number[], positions: number[], blockSize: number): { ids: number[]; positions: number[] } {
  if (ids.length <= blockSize) {
    return { ids, positions }
  }
  return {
    ids: ids.slice(-blockSize),
    positions: positions.slice(-blockSize),
  }
}

function appendToken(ids: number[], positions: number[], nextPosition: number, tokenId: number, blockSize: number): {
  ids: number[]
  positions: number[]
  nextPosition: number
} {
  const nextIds = [...ids, tokenId]
  const nextPositions = [...positions, nextPosition]
  const trimmed = trimContext(nextIds, nextPositions, blockSize)
  return {
    ids: trimmed.ids,
    positions: trimmed.positions,
    nextPosition: nextPosition + 1,
  }
}

export async function measureProvider(
  provider: ProviderMode,
  decodeCount = 10,
  mode: CacheMode = 'reuse',
): Promise<ProviderMeasurement> {
  const manifest = await loadManifest()
  const engine = new OnnxFlatLMEngine(manifest)
  await engine.load(MODEL_BASE_URL, provider)
  const built = buildSeedContext(manifest)
  let ids = built.ids
  let positions = built.positions
  let nextPosition = built.nextPosition
  let logits: Float32Array
  const decodeSteps: Array<{ step: number; token: string; ms: number }> = []
  let totalDecodeMs = 0
  let prefillTotalMs = 0

  const forcedTokens = ['action_0']

  if (mode === 'reuse') {
    const t0 = performance.now()
    logits = await engine.prefill(ids, positions)
    prefillTotalMs += performance.now() - t0
  } else {
    logits = new Float32Array()
  }

  for (let step = 0; step < decodeCount; step += 1) {
    if (mode === 'recompute') {
      const t0 = performance.now()
      logits = await engine.prefill(ids, positions)
      prefillTotalMs += performance.now() - t0
    }
    let tokenId: number
    let token: string
    if (step < forcedTokens.length) {
      token = forcedTokens[step]
      tokenId = manifest.vocab[token]
    } else {
      tokenId = sampleFromLogits(logits)
      token = manifest.id_to_token[String(tokenId)] ?? `id_${tokenId}`
    }
    const stepStart = performance.now()
    logits = await engine.step(tokenId, nextPosition)
    const ms = performance.now() - stepStart
    decodeSteps.push({ step: step + 1, token, ms: Math.round(ms * 1000) / 1000 })
    totalDecodeMs += ms
    const appended = appendToken(ids, positions, nextPosition, tokenId, manifest.model_config.block_size)
    ids = appended.ids
    positions = appended.positions
    nextPosition = appended.nextPosition
  }

  return {
    provider,
    mode,
    providersLabel: engine.providersLabel,
    promptTokens: ids.length,
    prefillMs: Math.round(prefillTotalMs * 1000) / 1000,
    decodeCount: decodeSteps.length,
    totalMs: Math.round((prefillTotalMs + totalDecodeMs) * 1000) / 1000,
    totalDecodeMs: Math.round(totalDecodeMs * 1000) / 1000,
    averageDecodeMs: Math.round((totalDecodeMs / decodeSteps.length) * 1000) / 1000,
    tokens: decodeSteps.map((item) => item.token),
    decodeSteps,
  }
}
