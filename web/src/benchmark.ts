import './style.css'

import { measureProvider, type CacheMode } from './measureRuntime.ts'

type Backend = 'webgpu' | 'wasm'
type Point = { x: number; y: number }

const DECODE_TOKENS = [1, 4, 8, 16]

const rerollButton = document.querySelector<HTMLButtonElement>('#reroll-button')!
const backendModeSelect = document.querySelector<HTMLSelectElement>('#backend-mode-select')!
const cacheBackendSelect = document.querySelector<HTMLSelectElement>('#cache-backend-select')!
const aboutToggle = document.querySelector<HTMLButtonElement>('#about-toggle')!
const aboutDialog = document.querySelector<HTMLDialogElement>('#about-dialog')!

const charts = {
  backendLatency: document.querySelector<SVGSVGElement>('#backend-latency-chart')!,
  backendThroughput: document.querySelector<SVGSVGElement>('#backend-throughput-chart')!,
  cacheLatency: document.querySelector<SVGSVGElement>('#cache-latency-chart')!,
  cacheThroughput: document.querySelector<SVGSVGElement>('#cache-throughput-chart')!,
}

type ScenarioKey = `${Backend}:${CacheMode}:${number}`

const scenarioCache = new Map<ScenarioKey, { totalMs: number; throughput: number }>()
let running = false
const DEFAULT_BUTTON_LABEL = 're-run'

function setStatus(text: string): void {
  rerollButton.textContent = text
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}

function formatTick(value: number): string {
  if (Math.abs(value) < 0.005) {
    return '0'
  }
  if (value >= 10 || Math.abs(value - Math.round(value)) < 0.005) {
    return String(Math.round(value))
  }
  return value.toFixed(2)
}

function niceStep(maxValue: number, tickCount: number): number {
  if (maxValue <= 0) {
    return 1
  }
  const rough = maxValue / tickCount
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const normalized = rough / magnitude
  let niceNormalized: number
  if (normalized <= 1) {
    niceNormalized = 1
  } else if (normalized <= 2.5) {
    niceNormalized = 2.5
  } else if (normalized <= 5) {
    niceNormalized = 5
  } else {
    niceNormalized = 10
  }
  return niceNormalized * magnitude
}

function niceMax(rawMax: number, tickCount: number): number {
  const step = niceStep(rawMax, tickCount)
  return Math.max(step, Math.ceil(rawMax / step) * step)
}

function throughput(totalMs: number, decodeTokens: number): number {
  return round2((decodeTokens * 1000) / totalMs)
}

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string>): SVGElementTagNameMap[K] {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag)
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value)
  }
  return element
}

function clearChart(svg: SVGSVGElement): void {
  svg.replaceChildren()
}

function renderLineChart(
  svg: SVGSVGElement,
  lines: { label: string; color: string; points: Point[] }[],
  yLabel: string,
): void {
  clearChart(svg)
  const width = 500
  const height = 210
  const left = 58
  const right = 16
  const top = 26
  const bottom = 34
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const allPoints = lines.flatMap((line) => line.points)
  const rawMaxY = Math.max(...allPoints.map((point) => point.y))
  const tickCount = 4
  const maxY = rawMaxY > 0 ? niceMax(rawMaxY * 1.08, tickCount - 1) : 1
  const maxX = Math.max(...allPoints.map((point) => point.x))

  svg.append(svgEl('rect', { x: '0', y: '0', width: String(width), height: String(height), fill: '#fffdf4' }))

  for (let i = 0; i < tickCount; i += 1) {
    const y = top + (plotHeight * i) / (tickCount - 1)
    const tickValue = maxY - (maxY * i) / (tickCount - 1)
    svg.append(svgEl('line', {
      x1: String(left),
      y1: String(y),
      x2: String(width - right),
      y2: String(y),
      stroke: 'rgba(17,17,17,0.15)',
      'stroke-width': '1',
    }))
    const label = svgEl('text', {
      x: String(left - 8),
      y: String(y + 4),
      'text-anchor': 'end',
      class: 'chart-axis-text',
    })
    label.textContent = formatTick(tickValue)
    svg.append(label)
  }

  svg.append(svgEl('line', {
    x1: String(left),
    y1: String(top + plotHeight),
    x2: String(width - right),
    y2: String(top + plotHeight),
    stroke: '#111',
    'stroke-width': '2',
  }))

  lines.forEach((line, lineIdx) => {
    const pathParts: string[] = []
    line.points.forEach((point, idx) => {
      const x = left + (point.x / maxX) * plotWidth
      const y = top + plotHeight - (point.y / maxY) * plotHeight
      pathParts.push(`${idx === 0 ? 'M' : 'L'}${x} ${y}`)
      svg.append(svgEl('circle', {
        cx: String(x),
        cy: String(y),
        r: '3.5',
        fill: line.color,
        stroke: '#111',
        'stroke-width': '1.5',
      }))
      if (lineIdx === 0) {
        const xLabel = svgEl('text', {
          x: String(x),
          y: String(top + plotHeight + 18),
          'text-anchor': 'middle',
          class: 'chart-axis-text',
        })
        xLabel.textContent = String(point.x)
        svg.append(xLabel)
      }
    })
    svg.append(svgEl('path', {
      d: pathParts.join(' '),
      fill: 'none',
      stroke: line.color,
      'stroke-width': '3',
    }))
  })

  const yText = svgEl('text', {
    x: '14',
    y: String(top + plotHeight / 2),
    transform: `rotate(-90 14 ${top + plotHeight / 2})`,
    class: 'chart-label-text',
  })
  yText.textContent = yLabel
  svg.append(yText)

  const xText = svgEl('text', {
    x: String(left + plotWidth / 2),
    y: String(height + 6),
    'text-anchor': 'middle',
    class: 'chart-label-text',
  })
  xText.textContent = 'decode tokens'
  svg.append(xText)

  const legendSpacing = 120
  const legendStartX = left + Math.max(0, (plotWidth - legendSpacing * (lines.length - 1) - 90) / 2)
  lines.forEach((line, idx) => {
    const x = legendStartX + idx * legendSpacing
    const y = 14
    svg.append(svgEl('line', {
      x1: String(x),
      y1: String(y),
      x2: String(x + 18),
      y2: String(y),
      stroke: line.color,
      'stroke-width': '3',
    }))
    const text = svgEl('text', {
      x: String(x + 26),
      y: String(y + 4),
      class: 'chart-legend-text',
    })
    text.textContent = line.label
    svg.append(text)
  })
}

async function ensureScenario(provider: Backend, mode: CacheMode, decodeCount: number): Promise<{ totalMs: number; throughput: number }> {
  const key: ScenarioKey = `${provider}:${mode}:${decodeCount}`
  const cached = scenarioCache.get(key)
  if (cached) {
    return cached
  }
  setStatus(`measuring ${provider} ${mode} ${decodeCount} tok…`)
  const result = await measureProvider(provider, decodeCount, mode)
  const value = {
    totalMs: result.totalMs,
    throughput: throughput(result.totalMs, decodeCount),
  }
  scenarioCache.set(key, value)
  return value
}

async function collectLine(provider: Backend, mode: CacheMode): Promise<{ latency: Point[]; throughput: Point[] }> {
  const latency: Point[] = []
  const throughputPoints: Point[] = []
  for (const decodeCount of DECODE_TOKENS) {
    const result = await ensureScenario(provider, mode, decodeCount)
    latency.push({ x: decodeCount, y: result.totalMs })
    throughputPoints.push({ x: decodeCount, y: result.throughput })
  }
  return { latency, throughput: throughputPoints }
}

async function render(): Promise<void> {
  if (running) {
    return
  }
  running = true
  rerollButton.setAttribute('aria-busy', 'true')
  scenarioCache.clear()

  const backendMode = backendModeSelect.value as CacheMode
  const cacheBackend = cacheBackendSelect.value as Backend

  const webgpuLine = await collectLine('webgpu', backendMode)
  const wasmLine = await collectLine('wasm', backendMode)
  renderLineChart(charts.backendLatency, [
    { label: 'webgpu', color: '#2f7ed8', points: webgpuLine.latency },
    { label: 'wasm', color: '#111', points: wasmLine.latency },
  ], 'ms')
  renderLineChart(charts.backendThroughput, [
    { label: 'webgpu', color: '#2f7ed8', points: webgpuLine.throughput },
    { label: 'wasm', color: '#111', points: wasmLine.throughput },
  ], 'tok/s')

  const reuseLine = await collectLine(cacheBackend, 'reuse')
  const recomputeLine = await collectLine(cacheBackend, 'recompute')
  renderLineChart(charts.cacheLatency, [
    { label: 'reuse kv', color: '#63d471', points: reuseLine.latency },
    { label: 'recompute', color: '#ff5a6f', points: recomputeLine.latency },
  ], 'ms')
  renderLineChart(charts.cacheThroughput, [
    { label: 'reuse kv', color: '#63d471', points: reuseLine.throughput },
    { label: 'recompute', color: '#ff5a6f', points: recomputeLine.throughput },
  ], 'tok/s')

  rerollButton.setAttribute('aria-busy', 'false')
  running = false
  setStatus(DEFAULT_BUTTON_LABEL)
}

rerollButton.addEventListener('click', () => {
  void render()
})
backendModeSelect.addEventListener('change', () => {
  void render()
})
cacheBackendSelect.addEventListener('change', () => {
  void render()
})
aboutToggle.addEventListener('click', () => {
  aboutDialog.showModal()
})

aboutDialog.showModal()

void render()
