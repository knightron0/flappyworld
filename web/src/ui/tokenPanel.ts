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
  const escaped = escapeHtml(token)
  if (token === 'action_1') {
    return `<span class="token-action-flap">${escaped}</span>`
  }
  if (token === 'done_1') {
    return `<span class="token-done">${escaped}</span>`
  }
  return escaped
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}
