/** KG search API service. */

import type { KGSearchResult } from '../types'
import { fetchJSON } from './api-client'

export function queryKGSearch(query: string, k: number = 10): Promise<KGSearchResult> {
  return fetchJSON<KGSearchResult>('/api/kg/query', {
    method: 'POST',
    body: JSON.stringify({ query, k }),
  })
}
