/** KG search API service. */

import type { KGSearchResult } from '../types'
import { fetchJSON } from './api-client'

export function queryKGSearch(
  query: string,
  k: number = 10,
  resolution?: number,
): Promise<KGSearchResult> {
  const body: Record<string, unknown> = { query, k }
  if (resolution !== undefined) body.resolution = resolution
  return fetchJSON<KGSearchResult>('/api/kg/query', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}
