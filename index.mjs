/**
 * sqlite-muninn — HNSW vector search + graph traversal + Node2Vec for SQLite
 *
 * ESM entry point. Works with better-sqlite3, node:sqlite (Node 22.5+), and bun:sqlite.
 */
import { readFileSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch } from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));

const EXT_MAP = { darwin: "dylib", linux: "so", win32: "dll" };

/** @returns {string} version string from VERSION file */
export const version = readFileSync(join(__dirname, "VERSION"), "utf8").trim();

/**
 * Get the path to the muninn loadable extension binary.
 *
 * When installed from git, the binary is at the repo root.
 * When installed from npm registry, it's in the platform-specific package.
 *
 * @returns {string} Absolute path to the extension binary
 */
export function getLoadablePath() {
  const ext = EXT_MAP[platform];
  if (!ext) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. ` +
        `Supported: ${Object.keys(EXT_MAP).join(", ")}`
    );
  }

  // Try repo root first (git install / local dev)
  const localPath = join(__dirname, `muninn.${ext}`);
  if (statSync(localPath, { throwIfNoEntry: false })) {
    return localPath;
  }

  // Try platform-specific package (npm registry install)
  const pkgName = `@sqlite-muninn/${platform}-${arch === "arm64" ? "arm64" : "x64"}`;
  try {
    const pkgPath = join(__dirname, "node_modules", pkgName, `muninn.${ext}`);
    if (statSync(pkgPath, { throwIfNoEntry: false })) {
      return pkgPath;
    }
  } catch {
    // Package not found
  }

  throw new Error(
    `muninn binary not found for ${platform}-${arch}. ` +
      `Build with 'make all' or install from npm: npm install sqlite-muninn`
  );
}

/**
 * Load the muninn extension into a SQLite database connection.
 *
 * @param {object} db - Database connection with loadExtension() method
 */
export function load(db) {
  db.loadExtension(getLoadablePath());
}
