/**
 * sqlite-muninn — HNSW vector search + graph traversal + Node2Vec for SQLite
 *
 * CommonJS entry point. Works with better-sqlite3, node:sqlite (Node 22.5+), and bun:sqlite.
 */
"use strict";

const { readFileSync, statSync } = require("node:fs");
const { join } = require("node:path");
const { platform, arch } = require("node:process");

const ROOT = join(__dirname, "..");

const EXT_MAP = { darwin: "dylib", linux: "so", win32: "dll" };

const version = readFileSync(join(ROOT, "VERSION"), "utf8").trim();

function getLoadablePath() {
  const ext = EXT_MAP[platform];
  if (!ext) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. ` +
        `Supported: ${Object.keys(EXT_MAP).join(", ")}`
    );
  }

  // Try repo root first (git install / local dev)
  const localPath = join(ROOT, `muninn.${ext}`);
  if (statSync(localPath, { throwIfNoEntry: false })) {
    return localPath;
  }

  // Try platform-specific package (npm registry install)
  const pkgName = `@sqlite-muninn/${platform}-${arch === "arm64" ? "arm64" : "x64"}`;
  try {
    const pkgPath = join(
      ROOT,
      "node_modules",
      pkgName,
      `muninn.${ext}`
    );
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

function load(db) {
  db.loadExtension(getLoadablePath());
}

module.exports = { version, getLoadablePath, load };
