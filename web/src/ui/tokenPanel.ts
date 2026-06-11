import type { TraceRecord } from '../types/manifest.ts'

type TokenGroup = {
  className: string
  label: string
  tokens: string[]
}

export class TokenPanel {
  private readonly root: HTMLElement

  constructor(root: HTMLElement) {
    this.root = root
  }

  render(records: TraceRecord[]): void {
    if (records.length === 0) {
      this.root.dataset.empty = 'true'
      this.root.textContent = 'Token stream appears here after the run starts.'
      return
    }
    delete this.root.dataset.empty
    this.root.innerHTML = records
      .map((record) => `<div class="token-line">${renderTokenLine(record.generated_tokens)}</div>`)
      .join('')
    this.root.scrollTop = this.root.scrollHeight
  }
}

function renderTokenLine(tokens: string[]): string {
  return groupTokens(tokens).map(renderGroup).join(' ')
}

function groupTokens(tokens: string[]): TokenGroup[] {
  const groups: TokenGroup[] = [
    { className: 'token-group-action', label: 'action token', tokens: [] },
    { className: 'token-group-bird', label: 'bird y position', tokens: [] },
    { className: 'token-group-pipe0', label: 'pipe 0 visibility, x position, and gap', tokens: [] },
    { className: 'token-group-pipe1', label: 'pipe 1 visibility, x position, and gap', tokens: [] },
    { className: 'token-group-status', label: 'pipe respawn and done flags', tokens: [] },
    { className: 'token-group-other', label: 'other tokens', tokens: [] },
  ]

  for (const token of tokens) {
    if (token.startsWith('action_')) {
      groups[0].tokens.push(token)
    } else if (token.startsWith('bird_y_')) {
      groups[1].tokens.push(token)
    } else if (token.startsWith('pipe0_')) {
      groups[2].tokens.push(token)
    } else if (token.startsWith('pipe1_')) {
      groups[3].tokens.push(token)
    } else if (token.startsWith('respawn_') || token.startsWith('done_')) {
      groups[4].tokens.push(token)
    } else {
      groups[5].tokens.push(token)
    }
  }

  return groups.filter((group) => group.tokens.length > 0)
}

function renderGroup(group: TokenGroup): string {
  return `<span class="token-group ${group.className}" data-label="${escapeHtml(group.label)}">${group.tokens
    .map(renderToken)
    .join(' ')}</span>`
}

function renderToken(token: string): string {
  const label = formatTokenLabel(token)
  const escaped = escapeHtml(label)
  if (token === 'action_1') {
    return `<span class="token-action-flap">${escaped}</span>`
  }
  if (token === 'done_1') {
    return `<span class="token-done">${escaped}</span>`
  }
  return escaped
}

function formatTokenLabel(token: string): string {
  if (token === 'action_1') {
    return 'FLAP'
  }
  if (token === 'action_0') {
    return 'IDLE'
  }
  if (token === 'respawn_1') {
    return 'RESPAWN'
  }
  if (token === 'respawn_0') {
    return 'NO_RESPAWN'
  }
  if (token === 'done_1') {
    return 'DONE'
  }
  if (token === 'done_0') {
    return 'ALIVE'
  }
  if (token === 'pipe0_present_1') {
    return 'P0_VISIBLE'
  }
  if (token === 'pipe0_present_0') {
    return 'P0_HIDDEN'
  }
  if (token === 'pipe1_present_1') {
    return 'P1_VISIBLE'
  }
  if (token === 'pipe1_present_0') {
    return 'P1_HIDDEN'
  }
  if (token === 'pipe0_x_hidden') {
    return 'P0_X_HIDDEN'
  }
  if (token === 'pipe0_gap_hidden') {
    return 'P0_GAP_HIDDEN'
  }
  if (token === 'pipe1_x_hidden') {
    return 'P1_X_HIDDEN'
  }
  if (token === 'pipe1_gap_hidden') {
    return 'P1_GAP_HIDDEN'
  }
  return token.toUpperCase().replace('PIPE0_', 'P0_').replace('PIPE1_', 'P1_')
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}
