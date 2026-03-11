/** TanStack Query hooks for KG search. */

import { useMutation } from '@tanstack/react-query'
import * as kgService from '@/lib/services/kg-service'

export function useKGSearch() {
  return useMutation({
    mutationFn: ({ query, k }: { query: string; k?: number }) => kgService.queryKGSearch(query, k),
  })
}
