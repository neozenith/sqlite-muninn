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
  book_id: number
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

export interface Section {
  id: number
  slug: string
  /** Stable display name for test labels */
  name: string
  /** Builds the URL to visit for this section given an optional database */
  pathFor: (db: ManifestDatabase | null) => string
  /** Which data-testid must be present for the page to count as "loaded" */
  loadedTestId: (db: ManifestDatabase | null) => string
  /** Whether this section requires a database */
  takesDatabase: boolean
}

export const SECTIONS: readonly Section[] = [
  {
    id: 0,
    slug: 'HOME',
    name: 'Home',
    pathFor: () => '/',
    loadedTestId: () => 'home-database-list',
    takesDatabase: false,
  },
  {
    id: 1,
    slug: 'DATABASE',
    name: 'Database',
    pathFor: (db) => {
      if (!db) throw new Error('DATABASE section requires a database')
      return `/${encodeURIComponent(db.id)}/`
    },
    loadedTestId: () => 'database-detail',
    takesDatabase: true,
  },
] as const

const pad = (n: number): string => String(n).padStart(2, '0')

export function screenshotSlug(section: Section, db: ManifestDatabase | null): string {
  const dbPart = db ? `D${pad(indexOfDb(db))}_${db.id.toUpperCase()}` : 'DNA'
  return `S${pad(section.id)}_${section.slug}-${dbPart}`
}

/** Stable index of a database in the manifest — drives the D{id} part of the slug */
function indexOfDb(db: ManifestDatabase): number {
  const all = loadManifest().databases
  return all.findIndex((d) => d.id === db.id)
}

/** Permutation = (section, database | null). Databases-less sections get one entry. */
export interface Permutation {
  section: Section
  database: ManifestDatabase | null
}

export function allPermutations(): Permutation[] {
  const manifest = loadManifest()
  const out: Permutation[] = []
  for (const section of SECTIONS) {
    if (section.takesDatabase) {
      for (const db of manifest.databases) {
        out.push({ section, database: db })
      }
    } else {
      out.push({ section, database: null })
    }
  }
  return out
}

export function permutationLabel(p: Permutation): string {
  const { section, database } = p
  const dbPart = database ? ` :: ${database.id}` : ''
  return `S${pad(section.id)}_${section.slug}${dbPart}`
}
