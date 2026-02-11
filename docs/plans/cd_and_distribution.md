# CD & Distribution Plan

> **Status:** Research / Proposal
> **Date:** 2026-02-12
> **Scope:** Publishing `muninn` to PyPI, NPM, GitHub Releases, and Homebrew
> **Depends on:** `ci_and_packaging.md` — all packaging structure must be in place first
> **Supersedes:** CD portions of `distribution_and_ci.md`

---

## Table of Contents

1. [Overview](#overview)
2. [Versioning & Release Flow](#versioning--release-flow)
3. [PyPI Publishing](#pypi-publishing)
4. [NPM Publishing](#npm-publishing)
5. [WASM Publishing](#wasm-publishing)
6. [GitHub Releases](#github-releases)
7. [Homebrew Formula](#homebrew-formula)
8. [Release Workflow (`release.yml`)](#release-workflow-releaseyml)
9. [Implementation Order](#implementation-order)

---

## Overview

CD automates **publishing pre-built binaries to registries**. This is optional — users can already install from git after CI/packaging is complete (see `ci_and_packaging.md`). Registry publishing provides:

- Faster installs (no compilation needed)
- Discoverability (searchable on PyPI/NPM)
- Version pinning without git SHAs
- Platform-specific binaries for users without a C toolchain

### What Gets Published

| Registry | Package(s) | Trigger |
|----------|-----------|---------|
| **PyPI** | `sqlite-muninn` (5 platform wheels + 1 sdist) | GitHub Release |
| **NPM** | `sqlite-muninn` (wrapper) + `@sqlite-muninn/{platform}` (5 packages) | GitHub Release |
| **NPM** | `sqlite-muninn-wasm` (1 universal package) | GitHub Release |
| **GitHub Releases** | `.so`, `.dylib`, `.dll`, amalgamation tarball | GitHub Release |
| **Homebrew** | `sqlite-muninn` formula (custom tap) | Manual after GitHub Release |

---

## Versioning & Release Flow

### Single Source of Truth

A `VERSION` file at the repo root. All build scripts read this:

```
0.1.0
```

### Semantic Versioning

- **MAJOR** — Breaking changes to SQL interface or behavior
- **MINOR** — New features (algorithms, TVFs, vtab columns)
- **PATCH** — Bug fixes, performance improvements

### Release Process

```
1. Bump VERSION file, commit:    echo "0.2.0" > VERSION && git add VERSION && git commit -m "release: v0.2.0"
2. Tag and push:                 git tag v0.2.0 && git push && git push --tags
3. Create GitHub Release:        (manually or via release-please)
4. release.yml triggers:
   a. Build on 4 CI runners (producing 5 platform binaries)
   b. Run tests on each platform
   c. Build amalgamation
   d. Build WASM
   e. Package Python wheels (5 platform-tagged)
   f. Package NPM packages (5 platform + 1 main + 1 WASM)
   g. Upload binaries + amalgamation to GitHub Release
   h. Publish to PyPI (trusted publisher / OIDC)
   i. Publish to NPM (trusted publisher + provenance)
```

### Multi-Package Version Coordination

All packages share the same version from `VERSION`:
- `sqlite_muninn/__init__.py` → `__version__`
- `npm/sqlite-muninn/package.json` → `"version"`
- All `@sqlite-muninn/{platform}/package.json` → `"version"`
- `npm/sqlite-muninn-wasm/package.json` → `"version"`
- Amalgamation header comment
- SKILL.md YAML frontmatter `{{VERSION}}`

Build scripts stamp the version from `VERSION` into all of these before packaging.

### Changelog Generation

Options:
1. **[git-cliff](https://github.com/orhun/git-cliff)** — generates from Conventional Commits
2. **[release-please](https://github.com/google-github-actions/release-please-action)** — creates release PRs
3. **Manual `CHANGELOG.md`** — simplest, used by most SQLite extensions

---

## PyPI Publishing

### Authentication

Use **PyPI Trusted Publishers** (OIDC, no API tokens):

1. Configure at `https://pypi.org/manage/project/sqlite-muninn/settings/publishing/`
2. Link to GitHub repo + workflow file + environment name
3. Publish with `pypa/gh-action-pypi-publish@release/v1`

### Wheel Building

Each CI runner produces a platform binary. A post-build script assembles platform-tagged wheels:

```bash
# scripts/build_wheels.py
# For each platform artifact:
# 1. Create wheel directory structure:
#    sqlite_muninn/__init__.py
#    sqlite_muninn/muninn.{so,dylib,dll}
#    sqlite_muninn/skills/muninn/SKILL.md
#    sqlite_muninn/skills/muninn/references/cookbook-python.md
#    sqlite_muninn/skills/muninn/references/vector-encoding.md
#    sqlite_muninn/skills/muninn/references/platform-caveats.md
# 2. Write METADATA, WHEEL, RECORD files
# 3. Tag: sqlite_muninn-{version}-py3-none-{platform}.whl
# 4. Zip into .whl
```

### Platform Wheel Tags

| Platform | Wheel Tag |
|----------|-----------|
| Linux x86_64 | `manylinux_2_17_x86_64` |
| Linux ARM64 | `manylinux_2_17_aarch64` |
| macOS (Universal) | `macosx_11_0_universal2` |
| Windows x86_64 | `win_amd64` |

**Note:** macOS ships a universal binary (arm64 + x86_64 via `lipo`), so a single `universal2` wheel covers both architectures.

### glibc Compatibility (Linux)

Building on `ubuntu-22.04` (glibc 2.35) means the `.so` requires glibc >= 2.35. For wider compatibility, build inside a manylinux container:

```yaml
container: quay.io/pypa/manylinux_2_28_x86_64  # glibc 2.28
```

For now, building directly on the runner is fine (matches sqlite-vec's approach). Revisit if users report compatibility issues.

### Publish Job

```yaml
publish-pypi:
  needs: build
  runs-on: ubuntu-latest
  environment: pypi
  permissions:
    id-token: write
  steps:
    - uses: actions/checkout@v4
    - uses: actions/download-artifact@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.13"
    - run: python scripts/build_wheels.py
    - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## NPM Publishing

### Package Architecture

The **esbuild pattern** — platform-specific optional dependencies:

```
@sqlite-muninn/darwin-arm64     # macOS Apple Silicon
@sqlite-muninn/darwin-x64       # macOS Intel
@sqlite-muninn/linux-x64        # Linux x86_64
@sqlite-muninn/linux-arm64      # Linux ARM64
@sqlite-muninn/win32-x64        # Windows x86_64
sqlite-muninn                   # Main wrapper (optionalDependencies → above)
```

npm/yarn/pnpm automatically installs **only** the matching platform package.

### Platform Package (`@sqlite-muninn/linux-x64/package.json`)

```json
{
  "name": "@sqlite-muninn/linux-x64",
  "version": "0.1.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "files": ["muninn.so"]
}
```

### Main Package (`sqlite-muninn/package.json`)

```json
{
  "name": "sqlite-muninn",
  "version": "0.1.0",
  "main": "index.cjs",
  "module": "index.mjs",
  "types": "index.d.ts",
  "files": ["index.mjs", "index.cjs", "index.d.ts", "skills/**/*.md"],
  "optionalDependencies": {
    "@sqlite-muninn/darwin-arm64": "0.1.0",
    "@sqlite-muninn/darwin-x64": "0.1.0",
    "@sqlite-muninn/linux-x64": "0.1.0",
    "@sqlite-muninn/linux-arm64": "0.1.0",
    "@sqlite-muninn/win32-x64": "0.1.0"
  }
}
```

### Authentication

Use **npm trusted publishing** (OIDC) with provenance attestations:

```yaml
permissions:
  id-token: write
steps:
  - run: npm publish --provenance --access public
```

Configure per-package at `https://www.npmjs.com/package/{name}/access`.

### Publish Job

```yaml
publish-npm:
  needs: build
  runs-on: ubuntu-latest
  permissions:
    id-token: write
  steps:
    - uses: actions/checkout@v4
    - uses: actions/download-artifact@v4
    - uses: actions/setup-node@v4
      with:
        node-version: "22"
        registry-url: https://registry.npmjs.org
    - name: Publish platform packages
      run: |
        for target in darwin-arm64 darwin-x64 linux-x64 linux-arm64 win32-x64; do
          npm publish npm/@sqlite-muninn/$target --provenance --access public
        done
    - name: Publish main package
      run: npm publish npm/sqlite-muninn --provenance --access public
```

---

## WASM Publishing

### Package

A single platform-independent NPM package:

```json
{
  "name": "sqlite-muninn-wasm",
  "version": "0.1.0",
  "type": "module",
  "files": ["muninn_sqlite3.js", "muninn_sqlite3.wasm"]
}
```

### Publish Job

```yaml
publish-wasm:
  needs: wasm-build
  runs-on: ubuntu-latest
  permissions:
    id-token: write
  steps:
    - uses: actions/checkout@v4
    - uses: actions/download-artifact@v4
      with:
        name: wasm
        path: npm/sqlite-muninn-wasm/
    - uses: actions/setup-node@v4
      with:
        node-version: "22"
        registry-url: https://registry.npmjs.org
    - run: npm publish npm/sqlite-muninn-wasm --provenance --access public
```

---

## GitHub Releases

Upload binaries and amalgamation as release assets:

```yaml
upload-release:
  needs: [build, amalgamation, wasm-build]
  runs-on: ubuntu-latest
  permissions:
    contents: write
  steps:
    - uses: actions/download-artifact@v4
    - uses: softprops/action-gh-release@v2
      with:
        files: |
          linux-x86_64/muninn.so
          linux-arm64/muninn.so
          macos-universal/muninn.dylib
          windows-x86_64/muninn.dll
          amalgamation/muninn-amalgamation.tar.gz
          wasm/muninn_sqlite3.js
          wasm/muninn_sqlite3.wasm
```

---

## Homebrew Formula

Custom tap for macOS users:

```ruby
class SqliteMuninn < Formula
  desc "HNSW + graph traversal + Node2Vec SQLite extension"
  homepage "https://github.com/user/sqlite-muninn"
  url "https://github.com/user/sqlite-muninn/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<sha256>"
  license "MIT"
  depends_on "sqlite"

  def install
    system "make", "all", "SQLITE_PREFIX=#{Formula["sqlite"].opt_prefix}"
    lib.install "muninn.dylib"
    include.install "src/muninn.h"
  end

  test do
    system Formula["sqlite"].opt_bin/"sqlite3", ":memory:",
           ".load #{lib}/muninn", "SELECT 1"
  end
end
```

Publish as `brew tap user/tap` first. Consider `homebrew-core` if adoption grows.

---

## Release Workflow (`release.yml`)

```yaml
name: Release
on:
  release:
    types: [published]

jobs:
  # ── Build All Platforms ──────────────────────────────────
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-22.04
            target: linux-x86_64
            ext: so
          - os: ubuntu-22.04-arm
            target: linux-arm64
            ext: so
          - os: macos-15
            target: macos-universal
            ext: dylib
          - os: windows-2022
            target: windows-x86_64
            ext: dll
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: make all
      - name: Test
        run: make test
      - uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.target }}
          path: muninn.${{ matrix.ext }}

  # ── Amalgamation ─────────────────────────────────────────
  amalgamation:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: make amalgamation
      - run: tar czf muninn-amalgamation.tar.gz -C dist muninn.c muninn.h
      - uses: actions/upload-artifact@v4
        with:
          name: amalgamation
          path: muninn-amalgamation.tar.gz

  # ── WASM Build ───────────────────────────────────────────
  wasm-build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: mymindstorm/setup-emsdk@v14
      - run: make amalgamation
      - run: bash scripts/build_wasm.sh
      - uses: actions/upload-artifact@v4
        with:
          name: wasm
          path: "dist/muninn_sqlite3.*"

  # ── Publish to PyPI ──────────────────────────────────────
  publish-pypi:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: python scripts/build_wheels.py
      - uses: pypa/gh-action-pypi-publish@release/v1

  # ── Publish to NPM ──────────────────────────────────────
  publish-npm:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          registry-url: https://registry.npmjs.org
      - name: Publish platform packages
        run: |
          for target in darwin-arm64 darwin-x64 linux-x64 linux-arm64 win32-x64; do
            npm publish npm/@sqlite-muninn/$target --provenance --access public
          done
      - name: Publish main package
        run: npm publish npm/sqlite-muninn --provenance --access public

  # ── Publish WASM to NPM ─────────────────────────────────
  publish-wasm:
    needs: wasm-build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: wasm
          path: npm/sqlite-muninn-wasm/
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          registry-url: https://registry.npmjs.org
      - run: npm publish npm/sqlite-muninn-wasm --provenance --access public

  # ── Upload to GitHub Release ─────────────────────────────
  upload-release:
    needs: [build, amalgamation, wasm-build]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            linux-x86_64/muninn.so
            linux-arm64/muninn.so
            macos-universal/muninn.dylib
            windows-x86_64/muninn.dll
            amalgamation/muninn-amalgamation.tar.gz
            wasm/muninn_sqlite3.js
            wasm/muninn_sqlite3.wasm
```

---

## Implementation Order

All items below depend on CI/packaging being in place first.

### Phase 1: Release Infrastructure

1. **Add `VERSION` file** — if not already added during CI phase
2. **Create `scripts/build_wheels.py`** — assembles platform-tagged wheels from CI artifacts
3. **Create `scripts/stamp_version.sh`** — stamps `VERSION` into all package manifests

### Phase 2: PyPI

4. **Register `sqlite-muninn` on PyPI** — claim the name
5. **Configure PyPI trusted publisher** — OIDC link to GitHub repo
6. **Add `publish-pypi` job** to `release.yml`
7. **Test the full flow** — Tag → Release → PyPI

### Phase 3: NPM

8. **Register `sqlite-muninn` and `@sqlite-muninn/*` on NPM** — claim the names
9. **Create NPM platform package scaffolding** — 5 platform `package.json` files
10. **Configure NPM trusted publishing** — per-package OIDC
11. **Add `publish-npm` job** to `release.yml`
12. **Test with `better-sqlite3` and `node:sqlite`**

### Phase 4: WASM Package

13. **Create `npm/sqlite-muninn-wasm/` package** — `package.json` + JS wrapper
14. **Add `publish-wasm` job** to `release.yml`
15. **Test in browser** — verify HNSW + graph TVFs work end-to-end

### Phase 5: GitHub Releases + Homebrew

16. **Add `upload-release` job** — binaries + amalgamation + WASM
17. **Create Homebrew formula** — custom tap at `user/homebrew-tap`
18. **Document `brew install` flow**

### Phase 6: Ecosystem (Deferred)

19. **vcpkg / Conan ports** — when demand warrants
20. **`sqlite-muninn-bundled`** — Python package with baked-in SQLite (deferred unless demand)
21. **Fuzz testing** — libFuzzer harnesses
22. **Performance regression tracking** — Bencher or github-action-benchmark

---

## Appendix: Prior Art

| Project | PyPI | NPM | Approach |
|---------|------|-----|----------|
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | `py3-none-{platform}` wheels | Platform optionalDeps | Gold standard |
| [sqlean.py](https://github.com/nalgeon/sqlean.py) | CPython replacement | N/A | Bundles custom SQLite |
| [esbuild](https://github.com/evanw/esbuild) | N/A | Platform optionalDeps | Pioneered the NPM pattern |
| [sql.js](https://github.com/sql-js/sql.js) | N/A | WASM | SQLite in WebAssembly |
| [QMD](https://github.com/tobi/qmd) | N/A | N/A | `skills/` + SKILL.md pattern |

### Key Tools

| Tool | Purpose |
|------|---------|
| [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) | PyPI trusted publishing |
| [softprops/action-gh-release](https://github.com/softprops/action-gh-release) | GitHub Release asset uploads |
| [git-cliff](https://github.com/orhun/git-cliff) | Changelog from Conventional Commits |
| [release-please](https://github.com/google-github-actions/release-please-action) | Automated release PRs |
