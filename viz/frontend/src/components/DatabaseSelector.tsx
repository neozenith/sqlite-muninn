import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { fetchDatabases, getStoredDatabaseId, selectDatabase } from '@/lib/services/db-service'

export function DatabaseSelector() {
  const queryClient = useQueryClient()
  const [switching, setSwitching] = useState(false)

  const { data } = useQuery({
    queryKey: ['databases'],
    queryFn: fetchDatabases,
    staleTime: 30_000,
  })

  // On mount, sync localStorage selection with server if they differ
  useEffect(() => {
    if (!data || data.databases.length === 0) return
    const stored = getStoredDatabaseId()
    if (stored && stored !== data.active && data.databases.some((db) => db.id === stored)) {
      setSwitching(true)
      selectDatabase(stored)
        .then(() => queryClient.invalidateQueries())
        .finally(() => setSwitching(false))
    }
  }, [data, queryClient])

  if (!data || data.databases.length === 0) return null

  const activeDb = data.databases.find((db) => db.id === data.active)

  const handleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value
    if (!id) return
    setSwitching(true)
    await selectDatabase(id)
    await queryClient.invalidateQueries()
    setSwitching(false)
  }

  return (
    <div className="flex items-center gap-2">
      <select
        data-testid="db-selector"
        value={data.active ?? ''}
        onChange={handleChange}
        disabled={switching}
        className="h-7 rounded-md border bg-background px-2 text-xs disabled:opacity-50"
        aria-label="Select database"
      >
        {!data.active && <option value="">Select database...</option>}
        {data.databases.map((db) => (
          <option key={db.id} value={db.id}>
            {db.label}
          </option>
        ))}
      </select>
      {switching && (
        <span data-testid="db-switching" className="text-xs text-muted-foreground">
          Switching...
        </span>
      )}
      {!switching && activeDb && (
        <Badge variant="outline" className="text-[10px]">
          {activeDb.model} {activeDb.dim}d
        </Badge>
      )}
    </div>
  )
}
