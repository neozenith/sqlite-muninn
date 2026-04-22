import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function Layout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--color-surface)] text-[var(--color-foreground)]">
      <Sidebar />
      <div className="flex-1 overflow-y-auto" data-testid="app-main">
        <Outlet />
      </div>
    </div>
  )
}
