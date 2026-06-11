import * as ort from 'onnxruntime-web'

import type { Manifest, StepTimings } from '../types/manifest.ts'
import { concatSeqDimTensor, tensorToFloat32Array, truncateSeqDimTensor } from './tensorUtils.ts'

export type InferenceTimings = Pick<
  StepTimings,
  'prefillMs' | 'decodeMs' | 'decodeCalls' | 'syncDecodeCalls' | 'cacheReused'
>

export type ProviderMode = 'auto' | 'webgpu' | 'wasm'

export class KVCache {
  keys: ort.Tensor[]
  values: ort.Tensor[]

  constructor(keys: ort.Tensor[], values: ort.Tensor[]) {
    this.keys = keys
    this.values = values
  }

  get seqLen(): number {
    if (this.keys.length === 0) {
      return 0
    }
    return Number(this.keys[0].dims[2])
  }

  truncateLeft(drop: number): KVCache {
    if (drop <= 0) {
      return this
    }
    return new KVCache(
      this.keys.map((tensor) => truncateSeqDimTensor(tensor, drop)),
      this.values.map((tensor) => truncateSeqDimTensor(tensor, drop)),
    )
  }

  trimToMaxSeqLen(maxSeqLen: number): KVCache {
    if (maxSeqLen <= 0 || this.seqLen <= maxSeqLen) {
      return this
    }
    return this.truncateLeft(this.seqLen - maxSeqLen)
  }

  append(delta: KVCache, maxSeqLen?: number): KVCache {
    if (this.keys.length !== delta.keys.length || this.values.length !== delta.values.length) {
      throw new Error(`KV cache layer mismatch: ${this.keys.length} != ${delta.keys.length}`)
    }
    const appended = new KVCache(
      this.keys.map((tensor, layer) => concatSeqDimTensor(tensor, delta.keys[layer])),
      this.values.map((tensor, layer) => concatSeqDimTensor(tensor, delta.values[layer])),
    )
    if (maxSeqLen === undefined || appended.seqLen <= maxSeqLen) {
      return appended
    }
    return appended.truncateLeft(appended.seqLen - maxSeqLen)
  }
}

class TensorScratch {
  int64BatchTensor(ids: number[]): ort.Tensor {
    const data = new BigInt64Array(ids.length)
    for (let i = 0; i < ids.length; i += 1) {
      data[i] = BigInt(ids[i])
    }
    return new ort.Tensor('int64', data, [1, ids.length])
  }

  int64SingleTensor(value: number): ort.Tensor {
    return new ort.Tensor('int64', new BigInt64Array([BigInt(value)]), [1, 1])
  }
}

export class OnnxFlatLMEngine {
  private prefillSession: ort.InferenceSession | null = null
  private decodeSession: ort.InferenceSession | null = null
  private cache: KVCache | null = null
  private lastLogits: Float32Array | null = null
  private readonly manifest: Manifest
  private readonly scratch = new TensorScratch()
  private stepTimings: InferenceTimings = {
    prefillMs: 0,
    decodeMs: 0,
    decodeCalls: 0,
    syncDecodeCalls: 0,
    cacheReused: false,
  }
  private activeProviders: string[] = []
  readonly nLayer: number
  readonly blockSize: number
  readonly vocabSize: number

  constructor(manifest: Manifest) {
    this.manifest = manifest
    this.nLayer = manifest.onnx.n_layer
    this.blockSize = manifest.model_config.block_size
    this.vocabSize = manifest.model_config.vocab_size
  }

  async load(modelBaseUrl = `${import.meta.env.BASE_URL}model`, providerMode: ProviderMode = 'auto'): Promise<void> {
    configureWasmEnv()
    const cacheKey = encodeURIComponent(this.manifest.model_cache_key ?? `${this.vocabSize}-${this.blockSize}-${this.nLayer}`)
    const prefillUrl = `${modelBaseUrl}/${this.manifest.onnx.prefill}?cache=${cacheKey}`
    const decodeUrl = `${modelBaseUrl}/${this.manifest.onnx.decode}?cache=${cacheKey}`
    if (providerMode === 'wasm') {
      await this.createSessions(prefillUrl, decodeUrl, ['wasm'])
      return
    }
    if (providerMode === 'webgpu') {
      await this.createSessions(prefillUrl, decodeUrl, [{ name: 'webgpu' }])
      return
    }
    try {
      await this.createSessions(prefillUrl, decodeUrl, [{ name: 'webgpu' }, { name: 'wasm' }])
    } catch {
      await this.createSessions(prefillUrl, decodeUrl, ['wasm'])
    }
  }

  private async createSessions(
    prefillUrl: string,
    decodeUrl: string,
    providers: ort.InferenceSession.SessionOptions['executionProviders'],
  ): Promise<void> {
    const options: ort.InferenceSession.SessionOptions = { executionProviders: providers }
    this.prefillSession = await ort.InferenceSession.create(prefillUrl, options)
    this.decodeSession = await ort.InferenceSession.create(decodeUrl, options)
    this.activeProviders = providers?.map((provider) =>
      typeof provider === 'string' ? provider : provider.name,
    ) ?? ['wasm']
  }

  get providersLabel(): string {
    return `${this.activeProviders.join(' → ')} · ${ort.env.wasm.numThreads} wasm threads`
  }

  get cacheSeqLen(): number {
    return this.cache?.seqLen ?? 0
  }

  resetStepTimings(): void {
    this.stepTimings = {
      prefillMs: 0,
      decodeMs: 0,
      decodeCalls: 0,
      syncDecodeCalls: 0,
      cacheReused: false,
    }
  }

  getStepTimings(): InferenceTimings {
    return { ...this.stepTimings }
  }

  markCacheReused(): void {
    this.stepTimings.cacheReused = true
  }

  resetCache(): void {
    this.cache = null
    this.lastLogits = null
  }

  get currentLogits(): Float32Array | null {
    return this.lastLogits
  }

  private setLastLogits(logits: Float32Array): Float32Array {
    this.lastLogits = logits
    return logits
  }

  async prefill(inputIds: number[], positionIds: number[]): Promise<Float32Array> {
    if (!this.prefillSession) {
      throw new Error('ONNX prefill session not loaded')
    }
    const t0 = performance.now()
    const outputs = await this.prefillSession.run({
      input_ids: this.scratch.int64BatchTensor(inputIds),
      position_ids: this.scratch.int64BatchTensor(positionIds),
    })
    this.cache = readFullCache(outputs, this.nLayer).trimToMaxSeqLen(this.blockSize)
    this.stepTimings.prefillMs = performance.now() - t0
    return this.setLastLogits(OnnxFlatLMEngine.logitsAtLast(outputs.logits as ort.Tensor))
  }

  async step(tokenId: number, positionId: number, sync = false): Promise<Float32Array> {
    if (!this.decodeSession || !this.cache) {
      throw new Error('decode step called before prefill')
    }
    const t0 = performance.now()
    const feeds: Record<string, ort.Tensor> = {
      input_ids: this.scratch.int64SingleTensor(tokenId),
      position_ids: this.scratch.int64SingleTensor(positionId),
    }
    for (let layer = 0; layer < this.nLayer; layer += 1) {
      feeds[`past_key_${layer}`] = this.cache.keys[layer]
      feeds[`past_value_${layer}`] = this.cache.values[layer]
    }
    const outputs = await this.decodeSession.run(feeds)
    this.cache = readDecodeCache(
      outputs,
      this.cache,
      this.nLayer,
      this.manifest.onnx.decode_cache_output,
      this.blockSize,
    )
    const elapsed = performance.now() - t0
    this.stepTimings.decodeMs += elapsed
    this.stepTimings.decodeCalls += 1
    if (sync) {
      this.stepTimings.syncDecodeCalls += 1
    }
    return this.setLastLogits(OnnxFlatLMEngine.logitsAtLast(outputs.logits as ort.Tensor))
  }

  static logitsAtLast(logits: ort.Tensor): Float32Array {
    const dims = logits.dims.map((d) => Number(d))
    const [, seq, vocab] = dims
    const data = tensorToFloat32Array(logits)
    const offset = (seq - 1) * vocab
    return data.subarray(offset, offset + vocab)
  }
}

function configureWasmEnv(): void {
  ort.env.wasm.numThreads = 1
  ort.env.wasm.simd = true
}

function readFullCache(outputs: ort.InferenceSession.OnnxValueMapType, nLayer: number): KVCache {
  const keys: ort.Tensor[] = []
  const values: ort.Tensor[] = []
  for (let layer = 0; layer < nLayer; layer += 1) {
    const key = outputs[`present_key_${layer}`]
    const value = outputs[`present_value_${layer}`]
    if (!(key instanceof ort.Tensor) || !(value instanceof ort.Tensor)) {
      throw new Error(`Missing KV outputs for layer ${layer}`)
    }
    keys.push(key)
    values.push(value)
  }
  return new KVCache(keys, values)
}

function readDeltaCache(outputs: ort.InferenceSession.OnnxValueMapType, nLayer: number): KVCache {
  const keys: ort.Tensor[] = []
  const values: ort.Tensor[] = []
  for (let layer = 0; layer < nLayer; layer += 1) {
    const key = outputs[`new_key_${layer}`]
    const value = outputs[`new_value_${layer}`]
    if (!(key instanceof ort.Tensor) || !(value instanceof ort.Tensor)) {
      throw new Error(`Missing new KV outputs for layer ${layer}`)
    }
    keys.push(key)
    values.push(value)
  }
  return new KVCache(keys, values)
}

function readDecodeCache(
  outputs: ort.InferenceSession.OnnxValueMapType,
  cache: KVCache,
  nLayer: number,
  decodeCacheOutput: 'full' | 'delta' | undefined,
  maxSeqLen: number,
): KVCache {
  if (decodeCacheOutput === 'delta') {
    return cache.append(readDeltaCache(outputs, nLayer), maxSeqLen)
  }
  if (outputs.new_key_0 instanceof ort.Tensor) {
    return cache.append(readDeltaCache(outputs, nLayer), maxSeqLen)
  }
  return readFullCache(outputs, nLayer).trimToMaxSeqLen(maxSeqLen)
}
