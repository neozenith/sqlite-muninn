import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Boxes, Network, Workflow, Globe, PanelLeftClose, PanelLeftOpen, Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/components/ThemeProvider'

const STORAGE_KEY = 'muninn-sidebar'

const NAV_ITEMS = [
  { to: '/embeddings/chunks_vec', label: 'Embeddings', icon: Boxes },
  { to: '/graph/edges', label: 'Graph', icon: Network },
  { to: '/kg/query', label: 'KG Query', icon: Workflow },
] as const

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(STORAGE_KEY) === 'collapsed')
  const { resolvedTheme, toggleTheme } = useTheme()

  const toggle = () => {
    const next = !collapsed
    setCollapsed(next)
    localStorage.setItem(STORAGE_KEY, next ? 'collapsed' : 'expanded')
  }

  return (
    <aside
      className={`flex flex-col shrink-0 border-r bg-sidebar text-sidebar-foreground transition-[width] duration-200 ${collapsed ? 'w-12' : 'w-48'}`}
    >
      <div className="flex items-center justify-between p-2 border-b border-sidebar-border">
        {!collapsed && <span className="text-sm font-semibold tracking-tight pl-1">muninn</span>}
        <Button variant="ghost" size="icon-xs" onClick={toggle} aria-label="Toggle sidebar" className="shrink-0">
          {collapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
        </Button>
      </div>

      <nav className="flex-1 flex flex-col gap-0.5 p-1.5">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
              }`
            }
          >
            <item.icon className="size-4 shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="p-1.5 border-t border-sidebar-border space-y-0.5">
        <a
          href="/wasm/index.html"
          className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
        >
          <Globe className="size-4 shrink-0" />
          {!collapsed && <span>WASM Demo</span>}
        </a>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className="w-full flex items-center gap-2 justify-start px-2 py-1.5"
        >
          {resolvedTheme === 'dark' ? <Sun className="size-4 shrink-0" /> : <Moon className="size-4 shrink-0" />}
          {!collapsed && <span className="text-xs">{resolvedTheme === 'dark' ? 'Light' : 'Dark'}</span>}
        </Button>
      </div>
    </aside>
  )
}
