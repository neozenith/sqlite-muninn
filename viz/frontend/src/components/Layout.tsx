import { Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { AppSidebar } from '@/components/AppSidebar'
import { DatabaseSelector } from '@/components/DatabaseSelector'
import { fetchJSON } from '@/lib/services/api-client'
import type { HealthStatus } from '@/lib/types'

export function Layout() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetchJSON<HealthStatus>('/api/health'),
    staleTime: 30_000,
  })

  return (
    <div className="flex h-screen">
      <AppSidebar />

      <div className="flex flex-col flex-1 min-w-0">
        <header className="border-b px-4 py-1.5 flex items-center gap-2 shrink-0">
          {health?.extension_loaded && (
            <Badge variant="secondary" className="text-[10px]">
              extension loaded
            </Badge>
          )}
          <DatabaseSelector />
          {health && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {!health.active_database && <span>{health.db_path.split('/').pop()}</span>}
              {health.hnsw_index_count != null && (
                <Badge variant="outline" className="text-[10px]">
                  {health.hnsw_index_count} indexes
                </Badge>
              )}
              {health.edge_table_count != null && (
                <Badge variant="outline" className="text-[10px]">
                  {health.edge_table_count} graphs
                </Badge>
              )}
            </div>
          )}
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto p-4">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
