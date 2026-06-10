import { FlatLMStepper } from '../game/flatLmStepper.ts'
import { OnnxFlatLMEngine } from '../model/onnxEngine.ts'
import type { FrameTokens, Manifest } from '../types/manifest.ts'
import type {
  InferenceWorkerMessage,
  InferenceWorkerResponse,
  WorkerInitMessage,
  WorkerResetMessage,
  WorkerStepMessage,
} from './protocol.ts'

let engine: OnnxFlatLMEngine | null = null
let stepper: FlatLMStepper | null = null

function post(response: InferenceWorkerResponse): void {
  self.postMessage(response)
}

async function loadManifest(modelBaseUrl: string): Promise<Manifest> {
  const response = await fetch(`${modelBaseUrl}/manifest.json?cache=${Date.now()}`, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`Failed to load manifest.json (${response.status})`)
  }
  const manifest = await response.json() as Manifest
  validateManifest(manifest)
  return manifest
}

function validateManifest(manifest: Manifest): void {
  const vocabSize = manifest.model_config.vocab_size
  const bad = Object.entries(manifest.vocab).find(([, id]) => id < 0 || id >= vocabSize)
  if (bad) {
    throw new Error(`manifest/model mismatch: ${bad[0]}=${bad[1]} outside vocab_size=${vocabSize}`)
  }
}

function seedFromManifest(source: Manifest): { frames: FrameTokens[]; present: [boolean, boolean][] | null } {
  const seed = source.default_seed
  if (!seed) {
    throw new Error('manifest.json is missing default_seed; re-export with --seed-data')
  }
  return {
    frames: seed.frames,
    present: seed.pipe_present,
  }
}

async function init(message: WorkerInitMessage): Promise<void> {
  const manifest = await loadManifest(message.modelBaseUrl)
  engine = new OnnxFlatLMEngine(manifest)
  await engine.load(message.modelBaseUrl, message.provider)
  const { frames, present } = seedFromManifest(manifest)
  stepper = new FlatLMStepper(manifest, engine, frames, present, {
    pipeGapPx: 100,
  })
  post({
    type: 'ready',
    playerVersion: manifest.player_version,
    providersLabel: engine.providersLabel,
    state: stepper.latestState(),
    traceRecords: stepper.traceRecords,
  })
}

async function step(message: WorkerStepMessage): Promise<void> {
  if (!stepper) {
    throw new Error('worker step called before init')
  }
  const result = await stepper.step(message.flap)
  post({
    type: 'stepResult',
    requestId: message.requestId,
    result,
    state: stepper.latestState(),
    traceRecords: stepper.traceRecords,
    timings: stepper.lastTimings,
  })
}

function reset(message: WorkerResetMessage): void {
  if (!stepper) {
    throw new Error('worker reset called before init')
  }
  stepper.reset()
  post({
    type: 'resetDone',
    requestId: message.requestId,
    state: stepper.latestState(),
    traceRecords: stepper.traceRecords,
  })
}

self.addEventListener('message', (event: MessageEvent<InferenceWorkerMessage>) => {
  const message = event.data
  void (async () => {
    try {
      if (message.type === 'init') {
        await init(message)
      } else if (message.type === 'step') {
        await step(message)
      } else if (message.type === 'reset') {
        reset(message)
      } else {
        self.close()
      }
    } catch (error) {
      post({
        type: 'error',
        requestId: 'requestId' in message ? message.requestId : undefined,
        message: error instanceof Error ? error.message : String(error),
      })
    }
  })()
})
