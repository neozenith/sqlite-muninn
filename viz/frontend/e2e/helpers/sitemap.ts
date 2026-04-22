/**
 * Sitemap enumeration for E2E permutation coverage.
 *
 * Reads `frontend/public/demos/manifest.json` at module load to build the
 * per-database permutation list. The manifest is the single source of
 * truth — adding a new database creates a new test automatically.
 */
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))

export interface ManifestDatabase {
  id: string
  /** Optional — absent for session-log / non-book demos like sessions_demo. */
  book_id?: number
  model: string
  dim: number
  file: string
  size_bytes: number
  label: string
}

interface Manifest {
  databases: ManifestDatabase[]
}

const MANIFEST_PATH = resolve(HERE, '..', '..', 'public', 'demos', 'manifest.json')

export function loadManifest(): Manifest {
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf-8')) as Manifest
}

/** A viz table variant — one entry per value of `:tableId` in a route. */
export interface TableVariant {
  id: number
  slug: string
}

const EMBED_TABLES: readonly TableVariant[] = [
  { id: 0, slug: 'CHUNKS' },
  { id: 1, slug: 'ENTITIES' },
] as const

const KG_TABLES: readonly TableVariant[] = [
  { id: 0, slug: 'BASE' },
  { id: 1, slug: 'ER' },
] as const

export interface Section {
  id: number
  slug: string
  /** Stable display name for test labels */
  name: string
  /** Which table variants this section fans out over (empty = no table axis) */
  tables: readonly TableVariant[]
  /** Builds the URL for (section, database, table) */
  pathFor: (db: ManifestDatabase | null, table: TableVariant | null) => string
  /** data-testid that must be present for the page to count as "loaded" */
  loadedTestId: string
  /** Per-section timeout override (ms). fcose layout on 6K-node KG needs ~30-60s. */
  loadTimeoutMs: number
  /** Whether this section requires a database id */
  takesDatabase: boolean
}

export const SECTIONS: readonly Section[] = [
  {
    id: 0,
    slug: 'HOME',
    name: 'Home',
    tables: [],
    pathFor: () => '/',
    loadedTestId: 'home-database-list',
    loadTimeoutMs: 10000,
    takesDatabase: false,
  },
  {
    id: 1,
    slug: 'DATABASE',
    name: 'Database',
    tables: [],
    pathFor: (db) => {
      if (!db) throw new Error('DATABASE section requires a database')
      return `/${encodeURIComponent(db.id)}/`
    },
    loadedTestId: 'database-detail',
    loadTimeoutMs: 10000,
    takesDatabase: true,
  },
  {
    id: 2,
    slug: 'EMBED',
    name: 'Embed',
    tables: EMBED_TABLES,
    pathFor: (db, table) => {
      if (!db || !table) throw new Error('EMBED section requires a database and a table')
      return `/${encodeURIComponent(db.id)}/embed/${table.slug.toLowerCase()}/`
    },
    loadedTestId: 'embed-canvas-ready',
    loadTimeoutMs: 30000,
    takesDatabase: true,
  },
  {
    id: 3,
    slug: 'KG',
    name: 'KnowledgeGraph',
    tables: KG_TABLES,
    pathFor: (db, table) => {
      if (!db || !table) throw new Error('KG section requires a database and a table')
      return `/${encodeURIComponent(db.id)}/kg/${table.slug.toLowerCase()}/`
    },
    // `kg-canvas-ready` fires after Cytoscape mounts + an initial grid
    // layout lands; fcose refinement continues in the background. That
    // keeps ready-semantics tight to "data rendered" rather than "physics
    // converged" — which would take minutes on 6K-node graphs. 120s
    // accommodates parallel workers thrashing the server with BC compute
    // for large ER graphs (3300_MiniLM ER = 8K cluster nodes).
    loadedTestId: 'kg-canvas-ready',
    loadTimeoutMs: 120000,
    takesDatabase: true,
  },
] as const

const pad = (n: number): string => String(n).padStart(2, '0')

function indexOfDb(db: ManifestDatabase): number {
  const all = loadManifest().databases
  return all.findIndex((d) => d.id === db.id)
}

export function screenshotSlug(
  section: Section,
  db: ManifestDatabase | null,
  table: TableVariant | null,
): string {
  const dbPart = db ? `D${pad(indexOfDb(db))}_${db.id.toUpperCase()}` : 'DNA'
  const tablePart = table ? `T${pad(table.id)}_${table.slug}` : null
  return tablePart
    ? `S${pad(section.id)}_${section.slug}-${dbPart}-${tablePart}`
    : `S${pad(section.id)}_${section.slug}-${dbPart}`
}

export interface Permutation {
  section: Section
  database: ManifestDatabase | null
  table: TableVariant | null
}

export function allPermutations(): Permutation[] {
  const manifest = loadManifest()
  const out: Permutation[] = []
  for (const section of SECTIONS) {
    if (!section.takesDatabase) {
      out.push({ section, database: null, table: null })
      continue
    }
    for (const db of manifest.databases) {
      if (section.tables.length === 0) {
        out.push({ section, database: db, table: null })
      } else {
        for (const table of section.tables) {
          out.push({ section, database: db, table })
        }
      }
    }
  }
  return out
}

export function permutationLabel(p: Permutation): string {
  const { section, database, table } = p
  const bits: string[] = [`S${pad(section.id)}_${section.slug}`]
  if (database) bits.push(database.id)
  if (table) bits.push(table.slug)
  return bits.join(' :: ')
}
