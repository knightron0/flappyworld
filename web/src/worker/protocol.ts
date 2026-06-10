import type { ProviderMode } from '../model/onnxEngine.ts'
import type { RenderState, StepResult, StepTimings, TraceRecord } from '../types/manifest.ts'

export type WorkerInitMessage = {
  type: 'init'
  modelBaseUrl: string
  provider: ProviderMode
}

export type WorkerStepMessage = {
  type: 'step'
  requestId: number
  flap: boolean
}

export type WorkerResetMessage = {
  type: 'reset'
  requestId: number
}

export type WorkerShutdownMessage = {
  type: 'shutdown'
}

export type InferenceWorkerMessage =
  | WorkerInitMessage
  | WorkerStepMessage
  | WorkerResetMessage
  | WorkerShutdownMessage

export type WorkerReadyMessage = {
  type: 'ready'
  playerVersion: string
  providersLabel: string
  state: RenderState
  traceRecords: TraceRecord[]
}

export type WorkerStepResultMessage = {
  type: 'stepResult'
  requestId: number
  result: StepResult
  state: RenderState
  traceRecords: TraceRecord[]
  timings: StepTimings | null
}

export type WorkerResetDoneMessage = {
  type: 'resetDone'
  requestId: number
  state: RenderState
  traceRecords: TraceRecord[]
}

export type WorkerErrorMessage = {
  type: 'error'
  requestId?: number
  message: string
}

export type InferenceWorkerResponse =
  | WorkerReadyMessage
  | WorkerStepResultMessage
  | WorkerResetDoneMessage
  | WorkerErrorMessage
