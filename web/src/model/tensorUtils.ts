import * as ort from 'onnxruntime-web'

/** Decode IEEE-754 binary16 (stored in lower 16 bits of a number). */
function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) >> 15
  const exponent = (bits & 0x7c00) >> 10
  const fraction = bits & 0x03ff
  if (exponent === 0) {
    return (sign ? -1 : 1) * 2 ** -14 * (fraction / 1024)
  }
  if (exponent === 0x1f) {
    return fraction ? Number.NaN : (sign ? -Infinity : Infinity)
  }
  return (sign ? -1 : 1) * 2 ** (exponent - 15) * (1 + fraction / 1024)
}

export function tensorToFloat32Array(tensor: ort.Tensor): Float32Array {
  if (tensor.type === 'float32') {
    return tensor.data as Float32Array
  }
  if (tensor.type === 'float16') {
    const raw = tensor.data as Uint16Array | Float32Array
    const out = new Float32Array(raw.length)
    for (let i = 0; i < raw.length; i += 1) {
      const bits = raw[i] & 0xffff
      out[i] = float16BitsToFloat32(bits)
    }
    return out
  }
  throw new Error(`Unsupported tensor type for float conversion: ${tensor.type}`)
}

export function truncateSeqDimTensor(tensor: ort.Tensor, drop: number): ort.Tensor {
  const dims = tensor.dims.map((d) => Number(d))
  const [batch, nHead, seq, headDim] = dims
  const newSeq = seq - drop
  if (newSeq <= 0) {
    const empty = tensor.type === 'float16' ? new Uint16Array() : new Float32Array()
    return new ort.Tensor(tensor.type, empty, [batch, nHead, 0, headDim])
  }
  const elementSize = tensor.type === 'float16' ? 2 : 4
  const outByteLen = batch * nHead * newSeq * headDim * elementSize
  const outBuffer = new ArrayBuffer(outByteLen)
  const raw = tensor.data as ArrayBufferView
  const src = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength)

  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < nHead; h += 1) {
      for (let t = 0; t < newSeq; t += 1) {
        const srcStart = (((b * nHead + h) * seq + (t + drop)) * headDim) * elementSize
        const dstStart = (((b * nHead + h) * newSeq + t) * headDim) * elementSize
        const sliceLen = headDim * elementSize
        new Uint8Array(outBuffer, dstStart, sliceLen).set(src.subarray(srcStart, srcStart + sliceLen))
      }
    }
  }

  if (tensor.type === 'float16') {
    return new ort.Tensor('float16', new Uint16Array(outBuffer), [batch, nHead, newSeq, headDim])
  }
  return new ort.Tensor('float32', new Float32Array(outBuffer), [batch, nHead, newSeq, headDim])
}

export function concatSeqDimTensor(left: ort.Tensor, right: ort.Tensor): ort.Tensor {
  if (left.type !== right.type) {
    throw new Error(`Cannot concatenate tensors with different types: ${left.type} != ${right.type}`)
  }
  const leftDims = left.dims.map((d) => Number(d))
  const rightDims = right.dims.map((d) => Number(d))
  const [batch, nHead, leftSeq, headDim] = leftDims
  const [rightBatch, rightNHead, rightSeq, rightHeadDim] = rightDims
  if (batch !== rightBatch || nHead !== rightNHead || headDim !== rightHeadDim) {
    throw new Error(`Cannot concatenate KV tensors with shapes ${leftDims.join('x')} and ${rightDims.join('x')}`)
  }
  const outSeq = leftSeq + rightSeq
  const elementSize = left.type === 'float16' ? 2 : 4
  const outByteLen = batch * nHead * outSeq * headDim * elementSize
  const outBuffer = new ArrayBuffer(outByteLen)
  const leftRaw = left.data as ArrayBufferView
  const rightRaw = right.data as ArrayBufferView
  const leftSrc = new Uint8Array(leftRaw.buffer, leftRaw.byteOffset, leftRaw.byteLength)
  const rightSrc = new Uint8Array(rightRaw.buffer, rightRaw.byteOffset, rightRaw.byteLength)
  const dst = new Uint8Array(outBuffer)

  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < nHead; h += 1) {
      const leftSrcStart = ((b * nHead + h) * leftSeq * headDim) * elementSize
      const rightSrcStart = ((b * nHead + h) * rightSeq * headDim) * elementSize
      const dstStart = ((b * nHead + h) * outSeq * headDim) * elementSize
      const leftLen = leftSeq * headDim * elementSize
      const rightLen = rightSeq * headDim * elementSize
      dst.set(leftSrc.subarray(leftSrcStart, leftSrcStart + leftLen), dstStart)
      dst.set(rightSrc.subarray(rightSrcStart, rightSrcStart + rightLen), dstStart + leftLen)
    }
  }

  if (left.type === 'float16') {
    return new ort.Tensor('float16', new Uint16Array(outBuffer), [batch, nHead, outSeq, headDim])
  }
  return new ort.Tensor('float32', new Float32Array(outBuffer), [batch, nHead, outSeq, headDim])
}
