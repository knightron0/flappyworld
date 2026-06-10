export type ModelConfig = {
  vocab_size: number
  block_size: number
  n_layer: number
  n_head: number
  n_embd: number
  dropout: number
  max_position?: number
  rope_theta?: number
  position_scheme?: string
}

export type TokenizerConfig = {
  bird_y_bins: number
  pipe_x_bins: number
  pipe_gap_bins: number
  respawn_threshold_bins: number
}

export type FrameTokens = {
  bird_y: number
  pipe0_x: number
  pipe0_gap: number
  pipe1_x: number
  pipe1_gap: number
  respawn: number
  done: number
  action: number
}

export type DefaultSeed = {
  format: string
  line: number
  state_index: number
  frames: FrameTokens[]
  pipe_present: [boolean, boolean][] | null
}

export type Manifest = {
  player_version: string
  model_cache_key?: string
  transition: string
  position_scheme: string
  model_config: ModelConfig
  tokenizer_config: TokenizerConfig
  vocab: Record<string, number>
  id_to_token: Record<string, string>
  onnx: {
    prefill: string
    decode: string
    decode_cache_output?: 'full' | 'delta'
    n_layer: number
  }
  history_size: number
  default_seed?: DefaultSeed
}

export type RenderState = {
  p0_x: number
  p0_top: number
  p0_bottom: number
  p1_x: number
  p1_top: number
  p1_bottom: number
  bird_y: number
  action: string | null
  reward: string | null
}

export type StepResult = {
  action: string
  done: boolean
  respawn: boolean
}

export type StepTimings = {
  prefillMs: number
  decodeMs: number
  decodeCalls: number
  syncDecodeCalls: number
  cacheReused: boolean
  logicMs: number
  renderMs: number
  totalMs: number
}

export type TraceRecord = {
  step: number
  input_action: number
  generated_tokens: string[]
  pipe_present: { pipe0: boolean; pipe1: boolean }
  frame: FrameTokens & { action: number }
  timings?: StepTimings
}
