import { Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@/components/ThemeProvider'
import { Layout } from '@/components/Layout'
import { EmbeddingsPage } from '@/pages/EmbeddingsPage'
import { GraphPage } from '@/pages/GraphPage'
import { KGQueryPage } from '@/pages/KGQueryPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/kg/query/" replace />} />
            <Route path="embeddings/" element={<Navigate to="/embeddings/chunks_vec/" replace />} />
            <Route path="embeddings/:dataset" element={<EmbeddingsPage />} />
            <Route path="graph/" element={<Navigate to="/graph/edges/" replace />} />
            <Route path="graph/:dataset" element={<GraphPage />} />
            <Route path="kg/" element={<Navigate to="/kg/query/" replace />} />
            <Route path="kg/query/" element={<KGQueryPage />} />
            <Route path="*" element={<Navigate to="/kg/query/" replace />} />
          </Route>
        </Routes>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
