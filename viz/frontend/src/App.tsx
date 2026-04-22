import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { DatabasePage } from './pages/DatabasePage'
import { EmbedPage } from './pages/EmbedPage'
import { HomePage } from './pages/HomePage'
import { KGPage } from './pages/KGPage'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/:databaseId/" element={<DatabasePage />} />
        <Route path="/:databaseId/embed/:tableId/" element={<EmbedPage />} />
        <Route path="/:databaseId/kg/:tableId/" element={<KGPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
