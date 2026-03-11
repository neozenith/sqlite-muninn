import { describe, it, expect } from 'vitest'

describe('kg-pipeline transforms', () => {
  it('module is importable', async () => {
    const mod = await import('../transforms/kg-pipeline')
    expect(mod).toBeDefined()
  })
})
