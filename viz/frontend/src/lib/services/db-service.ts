/** Database listing and switching service. */

import { fetchJSON } from '@/lib/services/api-client'
import type { DatabasesResponse } from '@/lib/types'

const STORAGE_KEY = 'muninn-selected-db'

export async function fetchDatabases(): Promise<DatabasesResponse> {
  return fetchJSON<DatabasesResponse>('/api/databases')
}

export async function selectDatabase(id: string): Promise<void> {
  await fetchJSON('/api/databases/select', {
    method: 'POST',
    body: JSON.stringify({ id }),
  })
  localStorage.setItem(STORAGE_KEY, id)
}

export function getStoredDatabaseId(): string | null {
  return localStorage.getItem(STORAGE_KEY)
}
