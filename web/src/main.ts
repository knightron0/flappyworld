import './style.css'

import { drawFrame } from './render/flappyCanvas.ts'
import type { ProviderMode } from './model/onnxEngine.ts'
import type { RenderState, StepResult, StepTimings, TraceRecord } from './types/manifest.ts'
import { TokenPanel } from './ui/tokenPanel.ts'
import type {
  InferenceWorkerMessage,
  InferenceWorkerResponse,
  WorkerStepResultMessage,
} from './worker/protocol.ts'

const SCALE = 2
const SHOW_GUIDES = false
const DEFAULT_TARGET_MODEL_FPS = 20
const DEFAULT_PROVIDER: ProviderMode = 'wasm'
const MODEL_BASE_URL = `${import.meta.env.BASE_URL}model`

const statusBar = document.querySelector<HTMLElement>('#status-bar')!
const canvas = document.querySelector<HTMLCanvasElement>('#game-canvas')!
const tokenPanelShell = document.querySelector<HTMLElement>('#token-panel-shell')!
const tokenPanelRoot = document.querySelector<HTMLElement>('#token-panel')!
const tokenRate = document.querySelector<HTMLElement>('#token-rate')!
const tokenToggle = document.querySelector<HTMLButtonElement>('#token-toggle')!
const aboutToggle = document.querySelector<HTMLButtonElement>('#about-toggle')!
const aboutDialog = document.querySelector<HTMLDialogElement>('#about-dialog')!
const ctx = canvas.getContext('2d')
if (!ctx) {
  throw new Error('Canvas 2D context unavailable')
}

const tokenPanel = new TokenPanel(tokenPanelRoot)
const worker = new Worker(new URL('./worker/inferenceWorker.ts', import.meta.url), { type: 'module' })

let latestState: RenderState | null = null
let traceRecords: TraceRecord[] = []
let pendingFlap = false
let paused = false
let ready = false
let started = false
let done = false
let inFlight = false
let dirty = false
let frameIdx = 0
let lastTraceCount = 0
let requestId = 0
let currentStepStartedAt = 0
let nextStepAt = 0
let lastRenderMs = 0
let latestResult: StepResult | null = null
let latestTimings: StepTimings | null = null
let providersLabel = ''
let tokenStreamVisible = false
const completedStepTimes: number[] = []

function targetModelFps(): number {
  return DEFAULT_TARGET_MODEL_FPS
}

function modelStepIntervalMs(): number {
  return 1000 / targetModelFps()
}

function providerMode(): ProviderMode {
  return DEFAULT_PROVIDER
}

function post(message: InferenceWorkerMessage): void {
  worker.postMessage(message)
}

function setStatus(text: string): void {
  statusBar.textContent = text
}

function actualStepFps(now: number): number {
  const windowMs = 2000
  while (completedStepTimes.length > 0 && now - completedStepTimes[0] > windowMs) {
    completedStepTimes.shift()
  }
  return completedStepTimes.length / (windowMs / 1000)
}

function formatTimings(timings: StepTimings, renderMs: number): string {
  const genCalls = timings.decodeCalls - timings.syncDecodeCalls
  const avgDecode = timings.decodeCalls > 0 ? timings.decodeMs / timings.decodeCalls : 0
  const cacheLabel = timings.cacheReused
    ? timings.prefillMs === 0
      ? 'cache hit'
      : 'cache partial'
    : 'cache miss'
  return [
    `total ${timings.totalMs.toFixed(0)}ms`,
    cacheLabel,
    `prefill ${timings.prefillMs.toFixed(0)}ms`,
    `sync ${timings.syncDecodeCalls}`,
    `gen ${genCalls}x${avgDecode.toFixed(1)}ms`,
    `render ${renderMs.toFixed(0)}ms`,
  ].join(' | ')
}

function renderGame(updatePanel: boolean): number {
  if (!latestState) {
    return 0
  }
  const t0 = performance.now()
  drawFrame(ctx!, latestState, frameIdx, SCALE, SHOW_GUIDES)
  if (updatePanel && tokenStreamVisible) {
    tokenPanel.render(traceRecords)
    updateTokenRate()
    lastTraceCount = traceRecords.length
  }
  return performance.now() - t0
}

function updateTokenRate(): void {
  const tokenCount = traceRecords.reduce(
    (sum, record) => sum + record.generated_tokens.length,
    0,
  )
  const tokensPerFrame = traceRecords.length > 0 ? tokenCount / traceRecords.length : 0
  tokenRate.textContent = `${(tokensPerFrame * actualStepFps(performance.now())).toFixed(1)} tok/s`
}

function setTokenStreamVisible(visible: boolean): void {
  tokenStreamVisible = visible
  tokenPanelShell.hidden = !visible
  tokenToggle.setAttribute('aria-pressed', String(visible))
  tokenToggle.textContent = visible ? 'tokens: on' : 'tokens: off'
  if (visible) {
    tokenPanel.render(traceRecords)
    updateTokenRate()
    lastTraceCount = traceRecords.length
  }
}

function updateRunStatus(): void {
  if (!ready) {
    return
  }
  const targetFps = targetModelFps()
  if (!started) {
    setStatus(`press Space to start · ${providersLabel} · target ${targetFps} fps`)
    return
  }
  if (done) {
    setStatus(`done - press R to reset · ${providersLabel} · target ${targetFps} fps`)
    return
  }
  if (paused) {
    setStatus(`paused · ${providersLabel} · target ${targetFps} fps`)
    return
  }
  if (!latestResult) {
    setStatus(`ready · ${providersLabel} · target ${targetFps} fps`)
    return
  }

  let status = latestResult.action
  if (latestResult.respawn) {
    status += ' respawn'
  }
  if (latestResult.done) {
    status += ' done'
  }
  if (latestTimings) {
    status += ` · ${formatTimings(latestTimings, lastRenderMs)}`
  }
  status += ` · ${actualStepFps(performance.now()).toFixed(1)} fps (target ${targetFps})`
  if (inFlight) {
    status += ' · inference pending'
  }
  setStatus(status)
}

function sendStep(now: number): void {
  if (!ready || paused || done || inFlight) {
    return
  }
  inFlight = true
  currentStepStartedAt = now
  requestId += 1
  const flap = pendingFlap
  pendingFlap = false
  post({ type: 'step', requestId, flap })
  nextStepAt = now + modelStepIntervalMs()
}

function sendManualStep(): void {
  if (!ready || done || inFlight) {
    return
  }
  sendStep(performance.now())
}

function startGame(): void {
  if (!ready || started || done || inFlight) {
    return
  }
  started = true
  nextStepAt = performance.now()
  dirty = true
}

function queueFlap(): void {
  pendingFlap = true
  setStatus('flap queued')
}

function resetGame(): void {
  if (!ready || inFlight) {
    return
  }
  requestId += 1
  inFlight = true
  post({ type: 'reset', requestId })
}

function bindControls(): void {
  window.addEventListener('keydown', (event) => {
    if (event.code === 'Space' || event.code === 'ArrowUp') {
      event.preventDefault()
      if (!started) {
        startGame()
      }
      queueFlap()
    } else if (event.key === 'p' || event.key === 'P') {
      paused = !paused
      updateRunStatus()
    } else if (event.key === 's' || event.key === 'S') {
      sendManualStep()
    } else if (event.key === 'r' || event.key === 'R') {
      resetGame()
    }
  })
  canvas.addEventListener('click', () => {
    if (!started) {
      return
    }
    queueFlap()
  })
  tokenToggle.addEventListener('click', () => {
    setTokenStreamVisible(!tokenStreamVisible)
  })
  aboutToggle.addEventListener('click', () => {
    showAboutDialog()
  })
  aboutDialog.addEventListener('click', (event) => {
    if (event.target === aboutDialog) {
      aboutDialog.close()
    }
  })
}

function showAboutDialog(): void {
  if (aboutDialog.open) {
    return
  }
  if (typeof aboutDialog.showModal === 'function') {
    aboutDialog.showModal()
  } else {
    aboutDialog.setAttribute('open', '')
  }
}

function handleStepResult(message: WorkerStepResultMessage): void {
  inFlight = false
  latestState = message.state
  traceRecords = message.traceRecords
  latestResult = message.result
  latestTimings = message.timings
  done = message.result.done
  frameIdx += 1
  completedStepTimes.push(performance.now())
  if (message.timings) {
    const roundTripMs = performance.now() - currentStepStartedAt
    message.timings.totalMs = Math.max(message.timings.totalMs, roundTripMs)
  }
  dirty = true
}

function handleWorkerMessage(message: InferenceWorkerResponse): void {
  if (message.type === 'ready') {
    ready = true
    providersLabel = message.providersLabel
    latestState = message.state
    traceRecords = message.traceRecords
    dirty = true
    setStatus(`press Space to start · ${message.playerVersion} · ${providersLabel} · target ${targetModelFps()} fps`)
  } else if (message.type === 'stepResult') {
    handleStepResult(message)
  } else if (message.type === 'resetDone') {
    inFlight = false
    pendingFlap = false
    started = false
    done = false
    frameIdx = 0
    lastTraceCount = 0
    latestResult = null
    latestTimings = null
    completedStepTimes.length = 0
    latestState = message.state
    traceRecords = message.traceRecords
    dirty = true
    setStatus(`press Space to start · ${providersLabel} · target ${targetModelFps()} fps`)
  } else {
    inFlight = false
    setStatus(`worker error: ${message.message}`)
  }
}

function renderLoop(now: number): void {
  if (ready && started && !paused && !done && !inFlight && now >= nextStepAt) {
    sendStep(now)
  }
  if (dirty) {
    lastRenderMs = renderGame(traceRecords.length !== lastTraceCount)
    dirty = false
    updateRunStatus()
  } else if (latestState) {
    renderGame(false)
  }
  window.requestAnimationFrame(renderLoop)
}

function showBootFailure(message: string): void {
  setTokenStreamVisible(true)
  statusBar.style.display = 'block'
  statusBar.innerHTML = `<span class="error-banner">Failed to start: ${message}</span>`
  tokenPanelRoot.innerHTML = `
    <p class="error-banner">Export model artifacts first:</p>
    <pre>python3 -m scripts.export_model export \\
  --checkpoint checkpoints/g_rope/best.pt \\
  --out-dir web/public/model

python3 -m scripts.export_manifest \\
  --checkpoint checkpoints/g_rope/best.pt \\
  --seed-data dataset/data_flat.jsonl \\
  --out web/public/model/manifest.json</pre>
  `
}

function boot(): void {
  canvas.width = 288 * SCALE
  canvas.height = 512 * SCALE
  setTokenStreamVisible(false)
  bindControls()
  showAboutDialog()
  worker.addEventListener('message', (event: MessageEvent<InferenceWorkerResponse>) => {
    handleWorkerMessage(event.data)
  })
  worker.addEventListener('error', (event) => {
    showBootFailure(event.message)
  })
  setStatus('loading model worker...')
  post({ type: 'init', modelBaseUrl: MODEL_BASE_URL, provider: providerMode() })
  window.requestAnimationFrame(renderLoop)
}

boot()
