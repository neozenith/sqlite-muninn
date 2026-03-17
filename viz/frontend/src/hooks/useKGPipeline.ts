/** TanStack Query hooks for KG search. */

import { useMutation } from '@tanstack/react-query'
import * as kgService from '@/lib/services/kg-service'

export function useKGSearch() {
  return useMutation({
    mutationFn: ({ query, k, resolution }: { query: string; k?: number; resolution?: number }) =>
      kgService.queryKGSearch(query, k, resolution),
  })
}
