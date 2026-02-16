# CD & Distribution Plan

> **Status:** Active — infrastructure in place, registry publishing not yet implemented
> **Date:** 2026-02-12 (updated 2026-02-16)
> **Scope:** Publishing `muninn` to PyPI, NPM, and GitHub Releases
> **Depends on:** CI pipeline (`.github/workflows/ci.yml`) — fully implemented

---

## Table of Contents

1. [Overview](#overview)
2. [Versioning & Release Flow](#versioning--release-flow)
3. [Local Build Verification (`make dist`)](#local-build-verification-make-dist)
4. [PyPI Publishing](#pypi-publishing)
5. [NPM Publishing](#npm-publishing)
6. [WASM Publishing](#wasm-publishing)
7. [GitHub Releases](#github-releases)
8. [Publish Workflow (`publish.yml`)](#publish-workflow-publishyml)
9. [Implementation Order](#implementation-order)

---

## Overview

CD automates **publishing pre-built binaries to registries**. This is optional — users can already install from git after CI/packaging is complete. Registry publishing provides:

- Faster installs (no compilation needed)
- Discoverability (searchable on PyPI/NPM)
- Version pinning without git SHAs
- Platform-specific binaries for users without a C toolchain

### What Gets Published

| Registry | Package(s) | Trigger |
|----------|-----------|---------|
| **PyPI** | `sqlite-muninn` (5 platform wheels + 1 sdist) | GitHub Release |
| **NPM** | `sqlite-muninn` (wrapper) + `@neozenith/sqlite-muninn-{platform}` (5 packages) | GitHub Release |
| **NPM** | `@neozenith/sqlite-muninn-wasm` (1 universal package) | GitHub Release |
| **GitHub Releases** | `.so`, `.dylib`, `.dll`, amalgamation tarball | GitHub Release |

---

## Versioning & Release Flow

### Single Source of Truth: `VERSION` file

A plain-text `VERSION` file at the repo root. Currently: `0.1.0-alpha.1`.

```
0.1.0-alpha.1
```

#### Why VERSION File (Not Git-Tag-Based Versioning)

After evaluating `setuptools-scm`, `python-semantic-release`, and `commitizen`, the VERSION file approach was chosen because:

1. **Universally readable** — `cat VERSION` works in Make, Python, shell, C preprocessor (`-DVERSION=\"$(cat VERSION)\"`), Node.js, and CI. No tool-specific API needed.
2. **No `.git` dependency** — setuptools-scm needs `.git` at build time. Source tarballs and shallow clones break version detection. The VERSION file always works.
3. **Decoupled from commit conventions** — commit-based tools require strict Conventional Commits for version calculation. The VERSION file allows manual milestone versions (e.g., `1.0.0`).
4. **Separation of concerns** — git-cliff handles changelog generation, the VERSION file handles version storage, and `version_stamp.py` handles propagation. Each tool does one thing.

The stamp script (`scripts/version_stamp.py`) propagates the VERSION into `npm/package.json` and `skills/muninn/SKILL.md` via explicit regex targets. Run `make version-stamp` to propagate, or `python scripts/version_stamp.py --check` in CI to detect drift.

### Semantic Versioning

- **MAJOR** — Breaking changes to SQL interface or behavior
- **MINOR** — New features (algorithms, TVFs, vtab columns)
- **PATCH** — Bug fixes, performance improvements

Pre-release suffixes follow SemVer: `0.1.0-alpha.1`, `0.1.0-beta.1`, `0.1.0-rc.1`.

### Changelog: git-cliff

**Decided:** [git-cliff](https://github.com/orhun/git-cliff) (configured in `cliff.toml`, installed as dev dependency in `pyproject.toml`).

```bash
make changelog           # Generate/update CHANGELOG.md from git history
make release             # Calculate next version from commits, stamp everything, update changelog
```

The `make release` target:
1. Calculates next version from commit history via `git cliff --bumped-version`
2. Writes it to `VERSION`
3. Runs `make version-stamp` to propagate to all package manifests
4. Runs `git cliff --bump -o CHANGELOG.md` to generate the changelog
5. Prints instructions for the commit + tag

### Release Process

```
1. make release                    # Bumps VERSION, stamps manifests, updates CHANGELOG.md
2. Review changes
3. git add -A && git commit -m "chore(release): v0.2.0"
4. git tag v0.2.0
5. git push && git push --tags
6. Create GitHub Release (manually or via gh cli)
7. publish.yml triggers automatically
```

### Multi-Package Version Coordination

All packages share the same version from `VERSION`:
- `pyproject.toml` → `dynamic = ["version"]` reads VERSION directly
- `npm/package.json` → stamped by `version_stamp.py`
- `skills/muninn/SKILL.md` → stamped by `version_stamp.py`
- Amalgamation header → stamped by `scripts/amalgamate.sh`
- C compile-time → injected via `-DMUNINN_VERSION=\"$(VERSION)\"` in Makefile

---

## Local Build Verification (`make dist`)

Before pushing a release, verify all artifacts build locally:

```bash
make dist
```

This produces:

```
dist/
    muninn.dylib              # Native extension binary
    muninn.c                  # Amalgamation source
    muninn.h                  # Amalgamation header
    python/
        sqlite_muninn-*.whl   # Python wheel
    npm/
        sqlite-muninn-*.tgz   # npm package tarball
```

The `make dist` target runs: `version-stamp` → `build` → `amalgamation` → `python -m build --wheel` → `npm pack`.

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

The **esbuild pattern** — platform-specific optional dependencies, all under the `@neozenith` scope:

```
@neozenith/sqlite-muninn-darwin-arm64   # macOS Apple Silicon
@neozenith/sqlite-muninn-darwin-x64     # macOS Intel
@neozenith/sqlite-muninn-linux-x64      # Linux x86_64
@neozenith/sqlite-muninn-linux-arm64    # Linux ARM64
@neozenith/sqlite-muninn-win32-x64      # Windows x86_64
sqlite-muninn                           # Main wrapper (optionalDependencies -> above)
```

npm/yarn/pnpm automatically installs **only** the matching platform package.

### Platform Package (`@neozenith/sqlite-muninn-linux-x64/package.json`)

```json
{
  "name": "@neozenith/sqlite-muninn-linux-x64",
  "version": "0.1.0-alpha.1",
  "os": ["linux"],
  "cpu": ["x64"],
  "files": ["muninn.so"]
}
```

### Main Package (`sqlite-muninn/package.json`)

```json
{
  "name": "sqlite-muninn",
  "version": "0.1.0-alpha.1",
  "main": "index.cjs",
  "module": "index.mjs",
  "types": "index.d.ts",
  "files": ["index.mjs", "index.cjs", "index.d.ts"],
  "optionalDependencies": {
    "@neozenith/sqlite-muninn-darwin-arm64": "0.1.0-alpha.1",
    "@neozenith/sqlite-muninn-darwin-x64": "0.1.0-alpha.1",
    "@neozenith/sqlite-muninn-linux-x64": "0.1.0-alpha.1",
    "@neozenith/sqlite-muninn-linux-arm64": "0.1.0-alpha.1",
    "@neozenith/sqlite-muninn-win32-x64": "0.1.0-alpha.1"
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
          npm publish npm/platforms/$target --provenance --access public
        done
    - name: Publish main package
      run: npm publish npm/ --provenance --access public
```

---

## WASM Publishing

### Package

A single platform-independent NPM package:

```json
{
  "name": "@neozenith/sqlite-muninn-wasm",
  "version": "0.1.0-alpha.1",
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
        path: npm/wasm/
    - uses: actions/setup-node@v4
      with:
        node-version: "22"
        registry-url: https://registry.npmjs.org
    - run: npm publish npm/wasm --provenance --access public
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

## Publish Workflow (`publish.yml`)

**Implemented:** [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml)

Triggered by `release: [published]`. Jobs:

| Job | Depends on | What it does |
|-----|-----------|--------------|
| `build` | — | Matrix build: Linux x86_64, Linux ARM64, Windows x86_64 |
| `build-macos` | — | macOS universal binary via `lipo` (arm64 + x86_64) |
| `amalgamation` | — | Single-file `muninn.c` + `muninn.h` tarball |
| `wasm-build` | — | WASM module via `make -C wasm dist` + Emscripten |
| `publish-pypi` | build, build-macos | Assemble platform wheels, publish via OIDC trusted publisher |
| `publish-npm` | build, build-macos | Publish `sqlite-muninn` + 5 `@neozenith/sqlite-muninn-*` platform packages |
| `publish-wasm` | wasm-build | Publish `@neozenith/sqlite-muninn-wasm` |
| `upload-release` | build, build-macos, amalgamation, wasm-build | Attach all binaries to GitHub Release |

Key implementation details vs. original plan:
- macOS is a **separate job** (not a matrix entry) because universal binary requires two builds + `lipo`
- WASM build uses `make -C wasm dist` (not `scripts/build_wasm.sh` which was removed)
- Release assets are renamed for uniqueness (e.g., `muninn-linux-x86_64.so`)

---

## Implementation Order

### Phase 1: Release Infrastructure (Done)

1. ~~**Add `VERSION` file**~~ — `0.1.0-alpha.1`
2. ~~**Create `scripts/version_stamp.py`**~~ — stamps VERSION into package manifests
3. ~~**Configure git-cliff**~~ — `cliff.toml`, `make changelog`, `make release`
4. ~~**Add `make dist`**~~ — local build verification

### Phase 2: PyPI

5. **Create `scripts/build_wheels.py`** — assembles platform-tagged wheels from CI artifacts
6. **Register `sqlite-muninn` on PyPI** — claim the name
7. **Configure PyPI trusted publisher** — OIDC link to GitHub repo
8. **Add `publish-pypi` job** to `publish.yml`
9. **Test the full flow** — Tag -> Release -> PyPI

### Phase 3: NPM + WASM

10. **Register `sqlite-muninn` (unscoped) and `@neozenith/sqlite-muninn-*` platform packages on NPM**
11. **Create NPM platform package scaffolding** — 5 platform `package.json` files in `npm/platforms/`
12. **Create `npm/wasm/` package** — `@neozenith/sqlite-muninn-wasm` `package.json` + JS wrapper
13. **Configure NPM trusted publishing** — per-package OIDC (native + WASM)
14. **Add `publish-npm` and `publish-wasm` jobs** to `publish.yml`
15. **Test with `better-sqlite3`, `node:sqlite`, and browser** — verify native + WASM end-to-end

### Phase 4: GitHub Releases

16. **Add `upload-release` job** — binaries + amalgamation + WASM

---

## Appendix: Prior Art

| Project | PyPI | NPM | Approach |
|---------|------|-----|----------|
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | `py3-none-{platform}` wheels | Platform optionalDeps | Gold standard |
| [sqlean.py](https://github.com/nalgeon/sqlean.py) | CPython replacement | N/A | Bundles custom SQLite |
| [esbuild](https://github.com/evanw/esbuild) | N/A | Platform optionalDeps | Pioneered the NPM pattern |
| [sql.js](https://github.com/sql-js/sql.js) | N/A | WASM | SQLite in WebAssembly |

### Key Tools

| Tool | Purpose |
|------|---------|
| [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) | PyPI trusted publishing |
| [softprops/action-gh-release](https://github.com/softprops/action-gh-release) | GitHub Release asset uploads |
| [git-cliff](https://github.com/orhun/git-cliff) | Changelog from Conventional Commits |

### Versioning Research Summary

Tools evaluated for version management:

| Tool | Verdict | Why |
|------|---------|-----|
| `setuptools-scm` | Rejected | Python-only, needs `.git` at build time, doesn't solve multi-language |
| `python-semantic-release` | Rejected | Overkill, overlaps with git-cliff for changelogs |
| `commitizen` | Considered | `version_files` is nice but just declarative config for same regex stamping |
| `npm version` | Rejected | Single-ecosystem, doesn't propagate to Python/C |
| **VERSION file + stamp script** | **Chosen** | Universal, explicit, auditable, works everywhere |
| **git-cliff `--bumped-version`** | **Chosen** | Automates version *calculation* from commits |
