import { Navigate, Route, Routes } from 'react-router-dom'
import { DatabasePage } from './pages/DatabasePage'
import { HomePage } from './pages/HomePage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/:databaseId/" element={<DatabasePage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
