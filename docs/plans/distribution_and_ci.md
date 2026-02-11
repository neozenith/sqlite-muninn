# Distribution & CI/CD Plan

> **Status:** Research / Proposal
> **Date:** 2026-02-11
> **Scope:** Publishing `sqlite-muninn` to PyPI and NPM with cross-platform CI/CD

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Python Distribution (PyPI)](#python-distribution-pypi)
3. [Node.js Distribution (NPM)](#nodejs-distribution-npm)
4. [Custom SQLite Build Option](#custom-sqlite-build-option)
5. [Cross-Compilation Mechanics](#cross-compilation-mechanics)
6. [C/C++ Dependency Distribution](#cc-dependency-distribution)
7. [Cross-Platform CI/CD](#cross-platform-cicd)
8. [SQLite Version Testing](#sqlite-version-testing)
9. [Release Automation](#release-automation)
10. [Agent-Ready Distribution (SKILL.md)](#agent-ready-distribution-skillmd)
11. [Recommended Implementation Order](#recommended-implementation-order)

---

## Executive Summary

The goal is to make `muninn` installable via `pip install sqlite-muninn` and `npm install sqlite-muninn`, with precompiled binaries for all major platforms. The approach follows the pattern established by **sqlite-vec** (Alex Garcia), which is the gold standard for SQLite extension distribution.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build system | Keep Makefile + add MSVC script | Standard for SQLite extensions; CMake is overkill for zero-dependency C11 |
| Python packaging | `py3-none-{platform}` wheels with precompiled binary | No Python C API dependency; binary loaded via `load_extension()` |
| NPM packaging | Platform-specific `optionalDependencies` pattern | Proven by esbuild, Prisma, sqlite-vec at massive scale |
| WASM target | Required — compile via Emscripten with SQLite statically linked | Enables browser usage; accepts SIMD performance loss as trade-off |
| SQLite vendoring | Vendor `sqlite3.h` + `sqlite3ext.h` | Ensures consistent API surface; avoids macOS system SQLite issues |
| Minimum SQLite | 3.9.0 (table-valued functions) | Required by `graph_tvf.c`; practically test from 3.21.0+ |
| Packaging tool | Manual (not sqlite-dist) | sqlite-dist is WIP; the wheel/npm structure is simple enough to DIY |

### Current Gaps

- **No CI beyond docs deployment** — no build, test, or release workflows
- **No cross-platform builds** — only macOS (Homebrew) is tested today
- **No Windows support** — Makefile uses Unix conventions
- **No vendored SQLite headers** — relies on system/Homebrew `sqlite3.h`
- **No sanitizer CI** — ASan/UBSan exist in `make debug` but aren't run in CI
- **No multi-version SQLite testing**

---

## Python Distribution (PyPI)

### Package Name

`sqlite-muninn` on PyPI → `import sqlite_muninn` in Python.

### Package Structure

```
sqlite_muninn/
    __init__.py      # loadable_path() + load() + version
    muninn.so     # Linux (or .dylib on macOS, .dll on Windows)
```

The binary sits alongside `__init__.py`. No sub-packages needed.

### API Surface

```python
# sqlite_muninn/__init__.py
import os
import sqlite3

__version__ = "0.1.0"
__version_info__ = tuple(__version__.split("."))

def loadable_path() -> str:
    """Return path to the muninn loadable extension (without file extension).

    SQLite's load_extension() automatically appends .so/.dylib/.dll.
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "muninn"))

def load(conn: sqlite3.Connection) -> None:
    """Load muninn into the given SQLite connection."""
    conn.load_extension(loadable_path())
```

User-facing usage:

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)

# Now HNSW, graph TVFs, and node2vec are available
```

### Wheel Tags

The key insight: since `muninn` is loaded via `sqlite3.load_extension()` (not Python's C extension mechanism), it does **not** link against `libpython`. The wheel tag is:

```
sqlite_muninn-0.1.0-py3-none-{platform}.whl
```

- `py3` — works with any Python 3 version
- `none` — no Python ABI dependency
- `{platform}` — platform-specific because it contains a native binary

### Target Platforms

| Platform | Wheel Tag | CI Runner |
|----------|-----------|-----------|
| Linux x86_64 | `manylinux_2_17_x86_64` | `ubuntu-22.04` |
| Linux ARM64 | `manylinux_2_17_aarch64` | `ubuntu-24.04-arm` |
| macOS ARM64 | `macosx_11_0_arm64` | `macos-14` |
| macOS x86_64 | `macosx_10_13_x86_64` | `macos-13` |
| Windows x86_64 | `win_amd64` | `windows-2022` |

### Publishing Method

Use **PyPI Trusted Publishers** (OIDC, no API tokens):

1. Configure at `https://pypi.org/manage/project/sqlite-muninn/settings/publishing/`
2. Link to GitHub repo + workflow file + environment name
3. Publish with `pypa/gh-action-pypi-publish@release/v1`

### macOS Caveat

Apple's system SQLite is compiled with `SQLITE_OMIT_LOAD_EXTENSION`, so `load_extension()` is disabled. Users must either:

- Install Python via Homebrew (`brew install python`) — links against Homebrew SQLite with extensions enabled
- Install `pysqlite3-binary` and use it instead of the built-in `sqlite3`

This must be documented prominently in README and package description.

### Precedent Projects

| Project | Strategy | Notes |
|---------|----------|-------|
| **sqlite-vec** | `py3-none-{platform}` wheels, `os.path.dirname(__file__)` | Gold standard; uses `sqlite-dist` to generate scaffolding |
| **sqliteai-vector** | `importlib.resources.files()` for binary location | Alternative pattern; cleaner separation |
| **sqlean.py** | Replaces Python's `sqlite3` module entirely | Most aggressive; bundles custom SQLite build |
| **pysqlite3** | Bundles SQLite amalgamation as CPython extension | For when you need a specific SQLite version |

---

## Node.js Distribution (NPM)

### Package Name

`sqlite-muninn` on NPM (main wrapper) + `@sqlite-muninn/{platform}` platform packages.

### Architecture: Platform-Specific Optional Dependencies

This is the **esbuild pattern**, also used by sqlite-vec, Prisma, and SWC:

```
npm/
  sqlite-muninn/                    # Main wrapper package
    package.json                # optionalDependencies → platform packages
    index.mjs                   # ESM entry
    index.cjs                   # CJS entry
    index.d.ts                  # TypeScript declarations
  @sqlite-muninn/
    darwin-arm64/               # macOS Apple Silicon
      package.json              # os: ["darwin"], cpu: ["arm64"]
      muninn.dylib
    darwin-x64/                 # macOS Intel
      package.json
      muninn.dylib
    linux-x64/                  # Linux x86_64
      package.json
      muninn.so
    linux-arm64/                # Linux ARM64
      package.json
      muninn.so
    win32-x64/                  # Windows x86_64
      package.json
      muninn.dll
```

### Platform Package Structure

Each platform package is minimal:

```json
{
  "name": "@sqlite-muninn/darwin-arm64",
  "version": "0.1.0",
  "os": ["darwin"],
  "cpu": ["arm64"],
  "files": ["muninn.dylib"]
}
```

npm/yarn/pnpm automatically installs **only** the matching platform package.

### Main Package

```json
{
  "name": "sqlite-muninn",
  "version": "0.1.0",
  "main": "index.cjs",
  "module": "index.mjs",
  "types": "index.d.ts",
  "optionalDependencies": {
    "@sqlite-muninn/darwin-arm64": "0.1.0",
    "@sqlite-muninn/darwin-x64": "0.1.0",
    "@sqlite-muninn/linux-x64": "0.1.0",
    "@sqlite-muninn/linux-arm64": "0.1.0",
    "@sqlite-muninn/win32-x64": "0.1.0"
  }
}
```

### API Surface

```typescript
// index.d.ts
export declare function getLoadablePath(): string;

interface Db {
  loadExtension(file: string, entrypoint?: string | undefined): void;
}

export declare function load(db: Db): void;
```

```javascript
// index.mjs
import { statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch } from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PLATFORM_MAP = {
  "darwin-arm64": { pkg: "@sqlite-muninn/darwin-arm64", ext: "dylib" },
  "darwin-x64":   { pkg: "@sqlite-muninn/darwin-x64",   ext: "dylib" },
  "linux-x64":    { pkg: "@sqlite-muninn/linux-x64",    ext: "so" },
  "linux-arm64":  { pkg: "@sqlite-muninn/linux-arm64",   ext: "so" },
  "win32-x64":    { pkg: "@sqlite-muninn/win32-x64",    ext: "dll" },
};

export function getLoadablePath() {
  const key = `${platform}-${arch}`;
  const target = PLATFORM_MAP[key];
  if (!target) {
    throw new Error(`Unsupported platform: ${key}. Supported: ${Object.keys(PLATFORM_MAP).join(", ")}`);
  }
  const loadablePath = join(__dirname, "..", target.pkg, `muninn.${target.ext}`);
  if (!statSync(loadablePath, { throwIfNoEntry: false })) {
    throw new Error(
      `muninn binary not found for ${key}. ` +
      `Ensure ${target.pkg} is installed (it should be an optionalDependency).`
    );
  }
  return loadablePath;
}

export function load(db) {
  db.loadExtension(getLoadablePath());
}
```

### Compatible SQLite Drivers

The `Db` interface is deliberately minimal — it works with:

| Driver | Usage |
|--------|-------|
| `better-sqlite3` | `const db = new Database(":memory:"); load(db);` |
| `node:sqlite` (Node 22.5+) | `const db = new DatabaseSync(":memory:", { allowExtension: true }); load(db);` |
| `bun:sqlite` | `const db = new Database(":memory:"); load(db);` |

### Publishing

Use **npm trusted publishing** (OIDC) with provenance attestations:

```yaml
permissions:
  id-token: write
steps:
  - run: npm publish --provenance --access public
```

Must configure trusted publishing per-package at `https://www.npmjs.com/package/{name}/access`.

---

## Custom SQLite Build Option

### When to Ship Your Own SQLite

| Scenario | Recommendation |
|----------|---------------|
| Users just want to `pip install` and go | Ship loadable extension only (default) |
| macOS users hit `SQLITE_OMIT_LOAD_EXTENSION` | Document Homebrew Python; suggest `pysqlite3-binary` |
| Users need specific SQLite version or flags | Offer a `sqlite-muninn-bundled` package with bundled SQLite |
| Browser/WASM usage | Requires statically linked SQLite+extension build |

### Bundled SQLite Package (Optional, Future)

If demand warrants it, create a `sqlite-muninn-bundled` package that bundles the SQLite amalgamation with `muninn` compiled in:

```python
# This would be a CPython extension (like pysqlite3/sqlean.py)
# Replaces the built-in sqlite3 module
import sqlite_muninn_bundled as sqlite3  # drop-in replacement with muninn baked in
```

**Implementation approach (from sqlean.py):**

1. Download SQLite amalgamation (`sqlite3.c` + `sqlite3.h`)
2. Compile with `muninn` statically linked via `SQLITE_EXTRA_INIT` / `sqlite3_auto_extension()`
3. Package as a CPython extension module (version-specific wheels: `cp312-cp312-{platform}`)
4. Requires `cibuildwheel` for cross-platform CPython extension building

**Recommended compile flags:**

```
-DSQLITE_ENABLE_LOAD_EXTENSION
-DSQLITE_ENABLE_FTS5
-DSQLITE_ENABLE_JSON1
-DSQLITE_ENABLE_RTREE
-DSQLITE_ENABLE_STAT4
-DSQLITE_ENABLE_UPDATE_DELETE_LIMIT
-DSQLITE_TEMP_STORE=3
-DSQLITE_USE_URI
-DSQLITE_DQS=0
-O2 -fPIC
```

**Verdict:** Defer this. The loadable extension approach covers 90% of use cases. The bundled SQLite approach adds significant complexity (CPython ABI coupling, cibuildwheel, larger wheels) for marginal benefit.

### WASM Build (Required)

Compile SQLite + `muninn` to WebAssembly via Emscripten for browser and edge runtime usage. Extensions cannot be dynamically loaded in WASM — they must be statically linked at compile time.

**Static registration entry point:**

```c
// src/sqlite3_wasm_extra_init.c
#include "sqlite3.h"
extern int sqlite3_muninn_init(sqlite3*, char**, const sqlite3_api_routines*);

int sqlite3_wasm_extra_init(const char *z) {
    return sqlite3_auto_extension((void(*)(void))sqlite3_muninn_init);
}
```

**Build process (Emscripten):**

```bash
# 1. Download SQLite amalgamation
wget https://www.sqlite.org/2025/sqlite-amalgamation-3510000.zip
unzip sqlite-amalgamation-3510000.zip

# 2. Compile SQLite + muninn → WASM
emcc -O2 -s WASM=1 -s EXPORTED_FUNCTIONS='["_sqlite3_open", ...]' \
     -DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_JSON1 \
     sqlite-amalgamation-3510000/sqlite3.c \
     dist/muninn.c \
     src/sqlite3_wasm_extra_init.c \
     -o dist/muninn_sqlite3.js
```

**NPM distribution:** Ship as a separate `sqlite-muninn-wasm` package (not bundled with the native packages). The WASM binary is platform-independent — a single package works everywhere:

```json
{
  "name": "sqlite-muninn-wasm",
  "version": "0.1.0",
  "files": ["muninn_sqlite3.js", "muninn_sqlite3.wasm"],
  "type": "module"
}
```

**Performance trade-off:** WASM builds lose the SIMD-accelerated distance functions in `vec_math.c`. HNSW search will be slower than native. Emscripten does support WASM SIMD (`-msimd128`), which can recover some performance for the vector math operations — this should be investigated during implementation.

**CI requirements:** The release workflow needs an Emscripten build job:

```yaml
build-wasm:
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
    - uses: mymindstorm/setup-emsdk@v14
    - run: make amalgamation
    - run: bash scripts/build_wasm.sh
    - uses: actions/upload-artifact@v4
      with: { name: wasm, path: "dist/muninn_sqlite3.*" }
```

---

## Cross-Compilation Mechanics

The key question: how do you actually produce `.so`, `.dylib`, and `.dll` files for 5 platform targets from GitHub Actions?

### Strategy: Native Builds on Each Runner (No Cross-Compilation Needed)

For a zero-dependency C11 library, the simplest approach is to compile natively on each platform's runner. GitHub Actions now provides free runners for **all five targets**:

| Target | Runner | Arch | Free? | Notes |
|--------|--------|------|-------|-------|
| Linux x86_64 | `ubuntu-22.04` | x86_64 | Yes | Primary target; glibc 2.35 |
| Linux ARM64 | `ubuntu-22.04-arm` | arm64 | Yes | Native ARM64, no emulation |
| macOS ARM64 | `macos-15` | arm64 | Yes | Apple Silicon (M1+) |
| macOS x86_64 | `macos-15-intel` | x86_64 | Yes | **Last Intel runner** — available until Aug 2027 |
| Windows x86_64 | `windows-2022` | x86_64 | Yes | Has MSVC + MinGW preinstalled |

Because every target has a native runner, **no cross-compilation is strictly necessary**. Each job just runs `make all` on its native runner.

### macOS: Cross-Compilation with `-arch` and Universal Binaries

Apple's Clang can cross-compile between x86_64 and arm64 on a single runner — both SDKs are always present. This means you can build **both architectures on a single ARM64 runner** and combine them into a universal (fat) binary:

```bash
# Build both architectures on a single macos-15 (ARM64) runner
cc -arch arm64  -O2 -std=c11 -fPIC -dynamiclib -undefined dynamic_lookup \
   -Isrc -o muninn_arm64.dylib src/*.c -lm

cc -arch x86_64 -O2 -std=c11 -fPIC -dynamiclib -undefined dynamic_lookup \
   -Isrc -o muninn_x86_64.dylib src/*.c -lm

# Combine into a single universal binary
lipo -create muninn_arm64.dylib muninn_x86_64.dylib -output muninn.dylib

# Verify both slices are present
lipo -info muninn.dylib
# Architectures in the fat file: x86_64 arm64
```

**Universal binaries are recommended** — they work on both Intel and Apple Silicon Macs without users knowing their architecture. This also eliminates the need for a second macOS runner, saving CI costs and complexity.

**Makefile support** — add an `ARCH` variable:

```makefile
ifeq ($(UNAME_S),Darwin)
    # Cross-compilation: ARCH=x86_64 or ARCH=arm64
    ifdef ARCH
        CFLAGS_BASE += -arch $(ARCH)
    endif
    CFLAGS_BASE += -mmacosx-version-min=11.0  # Minimum deployment target
endif
```

### Linux: ARM64 Options

**Option A: Native ARM64 runner (recommended)**

```yaml
build-linux-arm64:
  runs-on: ubuntu-22.04-arm    # Native ARM64, no emulation overhead
  steps:
    - uses: actions/checkout@v4
    - run: make all
```

**Option B: Cross-compiler on x86_64 (fallback)**

```yaml
build-linux-arm64:
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
    - run: sudo apt-get install -y gcc-aarch64-linux-gnu
    - run: make CC=aarch64-linux-gnu-gcc all
```

The Makefile already uses `CC ?= cc` and `$(CC)` throughout, so cross-compilation works with just `CC=aarch64-linux-gnu-gcc` — no Makefile changes needed. But since native ARM64 runners are free, Option A is simpler.

### Windows: MSVC via `ilammy/msvc-dev-cmd`

Windows requires a separate build command since the Makefile uses Unix conventions. Use MSVC (matching SQLite's own build):

```yaml
build-windows:
  runs-on: windows-2022
  steps:
    - uses: actions/checkout@v4
    - uses: ilammy/msvc-dev-cmd@v1    # Sets up cl.exe in PATH
    - name: Build
      shell: cmd
      run: |
        cl.exe /O2 /MT /W4 /LD /Isrc ^
          src\muninn.c src\hnsw_vtab.c src\hnsw_algo.c ^
          src\graph_tvf.c src\node2vec.c src\vec_math.c ^
          src\priority_queue.c src\id_validate.c ^
          /Fe:muninn.dll
```

| Flag | Purpose |
|------|---------|
| `/O2` | Optimize for speed |
| `/MT` | Static CRT — DLL has zero runtime dependencies |
| `/W4` | High warning level |
| `/LD` | Create DLL |
| `/Fe:` | Output filename |

**Why MSVC over MinGW?** SQLite itself is built with MSVC. Python's sqlite3 module loads MSVC-built extensions. `/MT` produces a fully self-contained DLL with no vcruntime dependency. sqlite-vec also uses MSVC.

### The glibc Compatibility Problem (Linux)

When you build on `ubuntu-22.04` (glibc 2.35), the resulting `.so` embeds versioned glibc symbol references. It will **refuse to load** on systems with glibc < 2.35 (e.g., RHEL 8 has glibc 2.28, Amazon Linux 2 has 2.26).

**For a SQLite extension, this is usually fine** — users running SQLite interactively tend to have modern systems. But if you need maximum compatibility:

| Build Environment | glibc | Compatible With |
|-------------------|-------|-----------------|
| `ubuntu-22.04` runner | 2.35 | Ubuntu 22.04+, RHEL 9+, Fedora 36+ |
| `quay.io/pypa/manylinux_2_28_x86_64` container | 2.28 | RHEL 8+, Ubuntu 20.04+, Debian 10+ |
| `quay.io/pypa/manylinux2014_x86_64` container | 2.17 | Everything from 2014+ |

To verify your binary's glibc requirements:

```bash
objdump -T muninn.so | grep GLIBC_ | sed 's/.*GLIBC_//' | sort -Vu
```

**Building in a manylinux container** (if needed):

```yaml
build-linux-x86_64:
  runs-on: ubuntu-22.04
  container: quay.io/pypa/manylinux_2_28_x86_64
  steps:
    - uses: actions/checkout@v4
    - run: make all   # Builds with glibc 2.28 compatibility
```

**sqlite-vec does NOT use manylinux containers** — they build directly on the runner. For now, building on `ubuntu-22.04` is the pragmatic choice.

### Recommended Release Build Matrix

```yaml
jobs:
  build-linux-x86_64:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: make all
      - uses: actions/upload-artifact@v4
        with: { name: linux-x86_64, path: muninn.so }

  build-linux-arm64:
    runs-on: ubuntu-22.04-arm
    steps:
      - uses: actions/checkout@v4
      - run: make all
      - uses: actions/upload-artifact@v4
        with: { name: linux-arm64, path: muninn.so }

  build-macos-universal:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - run: make ARCH=arm64 all
      - run: mv muninn.dylib muninn_arm64.dylib
      - run: make clean && make ARCH=x86_64 all
      - run: mv muninn.dylib muninn_x86_64.dylib
      - run: lipo -create muninn_arm64.dylib muninn_x86_64.dylib -output muninn.dylib
      - uses: actions/upload-artifact@v4
        with: { name: macos-universal, path: muninn.dylib }

  build-windows:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - uses: ilammy/msvc-dev-cmd@v1
      - shell: cmd
        run: cl.exe /O2 /MT /W4 /LD /Isrc src\*.c /Fe:muninn.dll
      - uses: actions/upload-artifact@v4
        with: { name: windows-x86_64, path: muninn.dll }
```

This uses **4 CI jobs** to produce **5 platform binaries** (the macOS job produces a universal binary containing both architectures).

---

## C/C++ Dependency Distribution

How should C/C++ consumers depend on `muninn`? Unlike Python/Node.js, the C ecosystem has no single dominant package manager. Instead, there are several parallel distribution channels, ordered here by priority for the SQLite extension ecosystem.

### Priority 1: Amalgamation (Single `.c` + `.h` File)

This is the **primary distribution format** for the C ecosystem — it's how SQLite itself, sqlite-vec, and most SQLite extensions are consumed.

**What it is:** All source files concatenated into a single `muninn.c`, plus a public API header `muninn.h`. Consumer just adds two files to their project.

**Consumer usage:**

```bash
# Download from GitHub release
wget https://github.com/user/sqlite-muninn/releases/download/v0.1.0/muninn-amalgamation.tar.gz
tar xf muninn-amalgamation.tar.gz

# Build as loadable extension
gcc -O2 -fPIC -shared muninn.c -o muninn.so -lm           # Linux
cc -O2 -fPIC -dynamiclib muninn.c -o muninn.dylib -lm     # macOS

# Or compile into their application with static linking
gcc -O2 myapp.c muninn.c -lsqlite3 -lm -o myapp
```

**Why amalgamation is preferred:**
- Zero build-system coupling — works with Make, CMake, Meson, Zig, anything
- Single-translation-unit compilation enables 5-10% better optimization (compiler can inline across module boundaries)
- Trivial vendoring — just copy two files
- This is exactly how SQLite itself distributes (130+ source files → single `sqlite3.c`)

**How to create the amalgamation:**

Option A — Simple shell script (`scripts/amalgamate.sh`):

```bash
#!/bin/bash
set -euo pipefail
VERSION=$(cat VERSION)
OUT=dist/muninn.c

mkdir -p dist
cat > "$OUT" <<HEADER
/* muninn amalgamation - v${VERSION}
 * Generated $(date -u +%Y-%m-%d)
 * https://github.com/user/sqlite-muninn
 */
HEADER

# Concatenate headers (removing internal #include "..." directives)
for f in src/vec_math.h src/priority_queue.h src/hnsw_algo.h \
         src/id_validate.h src/hnsw_vtab.h src/graph_tvf.h \
         src/node2vec.h src/muninn.h; do
    echo "/* ---- $f ---- */"
    grep -v '#include "' "$f"
done >> "$OUT"

# Concatenate implementations
for f in src/vec_math.c src/priority_queue.c src/hnsw_algo.c \
         src/id_validate.c src/hnsw_vtab.c src/graph_tvf.c \
         src/node2vec.c src/muninn.c; do
    echo "/* ---- $f ---- */"
    grep -v '#include "' "$f"
done >> "$OUT"

# Copy public header
cp src/muninn.h dist/

echo "Amalgamation: dist/muninn.c ($(wc -l < "$OUT") lines)"
```

Option B — Use [cwoffenden/combiner](https://github.com/cwoffenden/combiner) for recursive `#include` resolution:

```bash
python combiner/combine.py -r src -o dist/muninn.c src/muninn.c
```

**Makefile target:**

```makefile
amalgamation: dist/muninn.c dist/muninn.h   ## Create amalgamation

dist/muninn.c dist/muninn.h: $(SRC) $(wildcard src/*.h)
	bash scripts/amalgamate.sh
```

### Priority 2: Pre-compiled Binaries (GitHub Releases)

Ship `.so`/`.dylib`/`.dll` files as GitHub Release assets. Consumers download and use `sqlite3_load_extension()` directly. This is what the release workflow already produces.

### Priority 3: `make install` with pkg-config

The standard Unix library installation convention. Enables `pkg-config --cflags --libs muninn` for downstream Makefiles.

**`muninn.pc.in` template:**

```
prefix=@PREFIX@
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: muninn
Description: HNSW + Graph Traversal + Node2Vec SQLite Extension
Version: @VERSION@
Requires: sqlite3
Libs: -L${libdir} -lmuninn -lm
Cflags: -I${includedir}
```

**Makefile additions:**

```makefile
PREFIX  ?= /usr/local
LIBDIR   = $(PREFIX)/lib
INCLUDEDIR = $(PREFIX)/include
PKGCONFIGDIR = $(LIBDIR)/pkgconfig
DESTDIR ?=
VERSION  = $(shell cat VERSION 2>/dev/null || echo 0.0.0)

muninn.pc: muninn.pc.in
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(VERSION)|g' $< > $@

install: muninn$(EXT) muninn.pc                ## Install library, headers, pkg-config
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -d $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 muninn$(EXT) $(DESTDIR)$(LIBDIR)/
	install -m 644 src/muninn.h $(DESTDIR)$(INCLUDEDIR)/
	install -m 644 muninn.pc $(DESTDIR)$(PKGCONFIGDIR)/

uninstall:                                           ## Remove installed files
	rm -f $(DESTDIR)$(LIBDIR)/muninn$(EXT)
	rm -f $(DESTDIR)$(INCLUDEDIR)/muninn.h
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/muninn.pc
```

**Consumer usage:**

```bash
# Install
make install PREFIX=/usr/local

# Consume from another Makefile
CFLAGS += $(shell pkg-config --cflags muninn)
LDFLAGS += $(shell pkg-config --libs muninn)
```

### Priority 4: CMake FetchContent

For CMake-based consumers, provide a `CMakeLists.txt` that supports both `FetchContent` (build from source) and `find_package` (use installed version).

**`CMakeLists.txt` at repo root:**

```cmake
cmake_minimum_required(VERSION 3.14)
project(muninn VERSION 0.1.0 LANGUAGES C)

find_package(SQLite3 REQUIRED)

add_library(muninn
    src/muninn.c src/hnsw_vtab.c src/hnsw_algo.c
    src/graph_tvf.c src/node2vec.c src/vec_math.c
    src/priority_queue.c src/id_validate.c
)

target_include_directories(muninn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(muninn PUBLIC SQLite::SQLite3 PRIVATE m)
target_compile_features(muninn PUBLIC c_std_11)
set_target_properties(muninn PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Only build tests when this is the top-level project (not when consumed via FetchContent)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    enable_testing()
    add_subdirectory(test)
endif()

# Install rules (for find_package)
include(GNUInstallDirs)
install(TARGETS muninn EXPORT muninn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES src/muninn.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT muninn-targets
    FILE muninn-config.cmake
    NAMESPACE muninn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/muninn
)
```

**Consumer CMakeLists.txt:**

```cmake
include(FetchContent)
FetchContent_Declare(
    muninn
    GIT_REPOSITORY https://github.com/user/sqlite-muninn.git
    GIT_TAG        v0.1.0       # Pin to a release tag
)
FetchContent_MakeAvailable(muninn)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE muninn)
```

The `CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR` guard is critical — it prevents tests and dev targets from building when consumed as a dependency.

### Priority 5: Git Submodules

The oldest and simplest approach. Consumer adds the repo as a submodule:

```bash
git submodule add -b v0.1.0 https://github.com/user/sqlite-muninn.git vendor/muninn
```

Then in their Makefile:

```makefile
CFLAGS += -Ivendor/muninn/src
SRCS   += vendor/muninn/src/muninn.c vendor/muninn/src/hnsw_vtab.c ...
```

No changes needed to `muninn` — submodules just clone the repo at a pinned commit. But submodules are considered legacy; FetchContent or the amalgamation is preferred.

### Priority 6: Homebrew Formula (macOS)

```ruby
class VecGraph < Formula
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
    (lib/"pkgconfig").install "muninn.pc"
  end

  test do
    system Formula["sqlite"].opt_bin/"sqlite3", ":memory:",
           ".load #{lib}/muninn", "SELECT 1"
  end
end
```

Publish as a custom tap first (`brew tap user/tap`), then consider submitting to homebrew-core if adoption grows.

### Priority 7: vcpkg / Conan (Deferred)

These require a `CMakeLists.txt` (already covered in Priority 4) plus a port/recipe file. Both are well-suited for Windows-heavy ecosystems. Defer until there's demand.

### SQLite Header Detection (Autoconfiguration)

The current Makefile hard-codes the Homebrew path on macOS. A more robust detection chain:

```makefile
# Try pkg-config first, then Homebrew, then default system paths
SQLITE_CFLAGS ?= $(shell pkg-config --cflags sqlite3 2>/dev/null)
SQLITE_LIBS   ?= $(shell pkg-config --libs sqlite3 2>/dev/null)

ifeq ($(SQLITE_CFLAGS),)
    ifeq ($(UNAME_S),Darwin)
        SQLITE_PREFIX ?= $(shell brew --prefix sqlite 2>/dev/null || echo /usr/local)
        SQLITE_CFLAGS = -I$(SQLITE_PREFIX)/include
        SQLITE_LIBS   = -L$(SQLITE_PREFIX)/lib -lsqlite3
    else
        SQLITE_CFLAGS =
        SQLITE_LIBS   = -lsqlite3
    endif
endif
```

This gives consumers three ways to point at SQLite:

1. **pkg-config** (auto-detected) — works on most Linux systems
2. **Homebrew** (auto-detected on macOS) — current behavior, preserved
3. **Manual override** — `make all SQLITE_CFLAGS="-I/path/to/include" SQLITE_LIBS="-L/path/to/lib -lsqlite3"`

### Distribution Channel Summary

| Channel | Consumer Effort | Your Effort | Reach |
|---------|----------------|-------------|-------|
| **Amalgamation** | Download 2 files, compile | Low (script) | Widest — any build system |
| **GitHub Releases** | Download binary, `load_extension()` | Already done by release CI | Wide — runtime loaders |
| **`make install`** | `./configure && make install` | Low (Makefile targets) | Unix developers |
| **CMakeLists.txt** | `FetchContent_Declare(...)` | Medium | CMake users |
| **Homebrew** | `brew install sqlite-muninn` | Low (formula file) | macOS developers |
| **pip / npm** | `pip install` / `npm install` | Medium (covered above) | Python / JS developers |
| **vcpkg / Conan** | `vcpkg install sqlite-muninn` | Medium | Windows / cross-platform |

---

## Cross-Platform CI/CD

### Current State

Only one workflow exists: `.github/workflows/docs.yml` (documentation deployment).

**Missing entirely:**
- Build verification on push/PR
- Cross-platform compilation testing
- C unit tests in CI
- Python integration tests in CI
- Sanitizer (ASan/UBSan/MSan) checks
- Multi-architecture builds
- Release automation

### Proposed Workflow Structure

```
.github/workflows/
    ci.yml          # Build + test on every push/PR
    release.yml     # Build all platforms + publish on GitHub Release
    docs.yml        # (existing) Documentation deployment
```

### CI Workflow (`ci.yml`)

Triggered on push and PR to `main`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ── Build & Test (Primary Platforms) ────────────────────
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-22.04
            name: Linux x86_64
          - os: ubuntu-24.04-arm
            name: Linux ARM64
          - os: macos-14
            name: macOS ARM64
          - os: macos-13
            name: macOS x86_64
          - os: windows-2022
            name: Windows x86_64
    runs-on: ${{ matrix.os }}
    name: Build (${{ matrix.name }})
    steps:
      - uses: actions/checkout@v4
      - name: Build extension
        run: make all
      - name: Run C unit tests
        run: make test
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Run Python integration tests
        run: |
          pip install pytest
          make test-python

  # ── Sanitizers (Linux only) ─────────────────────────────
  sanitize:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        sanitizer:
          - { name: "ASan+UBSan", flags: "-fsanitize=address,undefined" }
          - { name: "MSan",       flags: "-fsanitize=memory" }
    name: Sanitize (${{ matrix.sanitizer.name }})
    steps:
      - uses: actions/checkout@v4
      - name: Build with sanitizers
        run: |
          CC=clang CFLAGS_EXTRA="${{ matrix.sanitizer.flags }} -g -O1 -fno-omit-frame-pointer" make test

  # ── SQLite Version Matrix ──────────────────────────────
  sqlite-compat:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        sqlite:
          - { version: "3.21.0", year: "2017" }   # Practical minimum
          - { version: "3.38.0", year: "2022" }   # vtab_in support
          - { version: "3.44.0", year: "2023" }   # xIntegrity
          - { version: "latest", year: ""     }   # Latest stable
    name: SQLite ${{ matrix.sqlite.version }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-sqlite@v1
        with:
          sqlite-version: ${{ matrix.sqlite.version }}
          sqlite-year: ${{ matrix.sqlite.year }}
      - name: Build and test
        run: make test
```

### Release Workflow (`release.yml`)

Triggered when a GitHub Release is published:

```yaml
name: Release
on:
  release:
    types: [published]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-22.04
            target: linux-x64
            ext: so
            wheel_plat: manylinux_2_17_x86_64
          - os: ubuntu-24.04-arm
            target: linux-arm64
            ext: so
            wheel_plat: manylinux_2_17_aarch64
          - os: macos-14
            target: darwin-arm64
            ext: dylib
            wheel_plat: macosx_11_0_arm64
          - os: macos-13
            target: darwin-x64
            ext: dylib
            wheel_plat: macosx_10_13_x86_64
          - os: windows-2022
            target: win32-x64
            ext: dll
            wheel_plat: win_amd64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Build extension
        run: make all
      - name: Run tests
        run: make test
      - uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.target }}
          path: muninn.${{ matrix.ext }}

  # ── Package & Publish to PyPI ──────────────────────────
  publish-pypi:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - name: Build wheels
        run: python scripts/build_wheels.py  # Assembles platform-tagged wheels
      - uses: pypa/gh-action-pypi-publish@release/v1

  # ── Package & Publish to NPM ───────────────────────────
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
          node-version: 22
          registry-url: https://registry.npmjs.org
      - name: Publish platform packages
        run: |
          for target in darwin-arm64 darwin-x64 linux-x64 linux-arm64 win32-x64; do
            npm publish npm/@sqlite-muninn/$target --provenance --access public
          done
      - name: Publish main package
        run: npm publish npm/sqlite-muninn --provenance --access public

  # ── Upload to GitHub Release ───────────────────────────
  upload-release:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            linux-x64/muninn.so
            linux-arm64/muninn.so
            darwin-arm64/muninn.dylib
            darwin-x64/muninn.dylib
            win32-x64/muninn.dll
```

### Build System Changes Needed

The current Makefile needs these additions for cross-platform CI:

1. **Windows MSVC support** — Add a `Makefile.msc` or `build_windows.bat`:
   ```bat
   cl /O2 /W3 /LD src\muninn.c src\hnsw_vtab.c src\hnsw_algo.c ^
      src\graph_tvf.c src\node2vec.c src\vec_math.c src\priority_queue.c ^
      src\id_validate.c /Fe:muninn.dll
   ```

2. **Vendor SQLite headers** — Add a `scripts/vendor.sh`:
   ```bash
   SQLITE_VERSION="3510000"
   SQLITE_YEAR="2025"
   wget "https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip"
   unzip -o sqlite-amalgamation-*.zip
   cp sqlite-amalgamation-*/sqlite3.h sqlite-amalgamation-*/sqlite3ext.h src/
   ```

3. **Remove Homebrew hard-dependency** — Make the macOS Homebrew path a fallback, not a requirement. The vendored headers should be the primary source.

---

## SQLite Version Testing

### Minimum Version: 3.9.0 (2015-10-14)

This is when **table-valued functions** (eponymous virtual tables) were introduced, which `graph_tvf.c` relies on. Practically, test from **3.21.0+** since that's the oldest version in any supported Linux distro from 2018+.

### Key API Version Timeline

| SQLite Version | Date | Relevant Feature |
|---|---|---|
| **3.9.0** | 2015-10 | Table-valued functions; eponymous-only virtual tables (`NULL xCreate`) |
| **3.20.0** | 2017-08 | `sqlite3_value_pointer()` / `sqlite3_bind_pointer()` (pointer passing) |
| **3.21.0** | 2017-10 | `NE`, `ISNOT`, `ISNOTNULL`, `ISNULL` constraint operators for vtabs |
| **3.26.0** | 2018-12 | Shadow table access restrictions (security hardening) |
| **3.38.0** | 2022-02 | `sqlite3_vtab_in()` for IN operator; LIMIT/OFFSET constraints |
| **3.44.0** | 2023-11 | `xIntegrity` method (vtab module version 4) |

### Compile-Time Options That Break Extensions

| Flag | Impact |
|------|--------|
| `SQLITE_OMIT_LOAD_EXTENSION` | **Disables all extension loading.** macOS system SQLite uses this! |
| `SQLITE_OMIT_VIRTUALTABLE` | Disables virtual tables entirely. HNSW vtab and graph TVFs will not work. |

### Recommended CI Matrix

```yaml
sqlite-compat:
  strategy:
    matrix:
      sqlite:
        - { version: "3.21.0", year: "2017" }   # Oldest distro version still in use
        - { version: "3.38.0", year: "2022" }   # vtab_in, LIMIT/OFFSET
        - { version: "3.44.0", year: "2023" }   # xIntegrity
        - { version: "3.51.0", year: "2025" }   # Latest stable
```

Use [`actions/setup-sqlite@v1`](https://github.com/marketplace/actions/setup-sqlite-environment) for version management in CI.

### Conditional Feature Support

For features that depend on newer SQLite versions, use version guards:

```c
#if SQLITE_VERSION_NUMBER >= 3038000
  /* Use sqlite3_vtab_in() for efficient IN handling */
#else
  /* Fall back to sequential constraint evaluation */
#endif
```

---

## Release Automation

### Versioning

Semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR** — Breaking changes to SQL interface or behavior
- **MINOR** — New features (algorithms, TVFs, vtab columns)
- **PATCH** — Bug fixes, performance improvements

### Recommended Release Flow

```
1. Developer pushes a tag:     git tag v0.2.0 && git push --tags
2. GitHub Release is created:  (manually or via release-please)
3. release.yml triggers:
   a. Build on 5 platforms in parallel
   b. Run tests on each platform
   c. Package Python wheels (5 platform-tagged wheels)
   d. Package NPM packages (5 platform + 1 main)
   e. Upload binaries to GitHub Release
   f. Publish to PyPI (trusted publisher)
   g. Publish to NPM (trusted publisher + provenance)
```

### Changelog Generation

Options (in order of recommendation):

1. **[git-cliff](https://github.com/orhun/git-cliff)** — Generates changelogs from Conventional Commits
2. **[release-please](https://github.com/google-github-actions/release-please-action)** — Creates release PRs with auto-generated changelogs
3. **Manual `CHANGELOG.md`** — Simplest, used by most SQLite extension projects

### Multi-Package Version Coordination

All packages (Python wheel, NPM main, NPM platform packages) must share the same version. A `VERSION` file at the repo root can be the single source of truth:

```
0.1.0
```

Build scripts read this file and stamp it into `__init__.py`, `package.json`, and wheel metadata.

---

## Agent-Ready Distribution (SKILL.md)

### Inspiration: QMD's Skills Pattern

The `skills/` directory structure is inspired by [QMD](https://github.com/tobi/qmd) (Tobi Lütke), which pioneered a clean pattern for making CLI tools discoverable by AI coding assistants. QMD's approach:

```
skills/
    qmd/
        SKILL.md              # YAML frontmatter + structured usage docs
        references/
            mcp-setup.md      # Deep-dive supplementary docs
.claude-plugin/
    marketplace.json          # Plugin manifest (MCP servers, metadata)
```

Key design choices we're adopting from QMD:

| QMD Pattern | What It Does | Our Adaptation |
|-------------|-------------|----------------|
| **YAML frontmatter** | Machine-parseable metadata (name, description, triggers, `allowed-tools`) | Same — gives AI tools structured context before reading prose |
| **`references/` subdirectory** | Deep-dive docs alongside the skill | Per-language cookbooks as reference files |
| **Trigger phrases in description** | Tells AI *when* to activate (`"search my notes"`, `"find in docs"`) | Trigger on `"vector search"`, `"graph traversal"`, `"HNSW"`, `"knowledge graph"` |
| **`allowed-tools` frontmatter** | Declares what tools the skill needs | Scopes to SQL execution, Python/Node runtime |
| **`.claude-plugin/marketplace.json`** | Formal plugin registration for Claude Code ecosystem | Adopt for Claude Code plugin distribution |
| **Dynamic status checks** (`!` backtick) | Inline shell execution for live environment checks | Detect if muninn is installed / extension loadable |

### The Problem: AI Coding Tools Can't Use What They Can't Discover

When a developer asks Claude Code, Cursor, Copilot, or Aider to "add vector search to my SQLite database", the AI needs to:

1. **Know `muninn` exists** — discoverability
2. **Know how to install it** — package name, platform caveats
3. **Know the SQL interface** — `CREATE VIRTUAL TABLE ... USING hnsw_index(...)`, not some invented syntax
4. **Know the idiomatic patterns** — vector encoding, search queries, graph traversal, Node2Vec
5. **Know the gotchas** — macOS `load_extension` disabled, vector blob format, dimension constraints

Today, AI tools piece this together from README fragments, Stack Overflow, and hallucinated guesses. The result: incorrect SQL syntax, wrong vector encoding, missing `enable_load_extension()` calls, and frustrated developers who blame the extension instead of the AI.

**The fix:** Ship a structured `skills/` directory with a `SKILL.md` file and per-language reference cookbooks — a machine-readable knowledge base that AI coding tools can consume directly.

### Repository Structure

Following the QMD pattern, the skill definition lives in the repo and gets distributed with every package:

```
skills/
    muninn/
        SKILL.md                      # Main skill: YAML frontmatter + usage guide
        references/
            cookbook-python.md         # Python patterns (semantic search, RAG, batch insert)
            cookbook-node.md           # Node.js patterns (Express, Bun, buffer encoding)
            cookbook-c.md              # C/C++ patterns (static linking, blob binding)
            cookbook-sql.md            # Pure SQL patterns (CLI workflows, graph pipelines)
            vector-encoding.md        # Cross-language vector format reference
            platform-caveats.md       # macOS load_extension, Windows DLL, glibc compat
.claude-plugin/
    marketplace.json                  # Claude Code plugin manifest
```

### SKILL.md with YAML Frontmatter

The `SKILL.md` follows the QMD convention — YAML frontmatter for machine-parseable metadata, followed by structured usage documentation:

````markdown
---
name: muninn
description: >
  Add HNSW vector similarity search, graph traversal (BFS, DFS, shortest path,
  PageRank, connected components), and Node2Vec embedding generation to any SQLite
  database. Use when users need vector search, knowledge graphs, graph algorithms,
  semantic search, or RAG retrieval in SQLite. Triggers on "vector search",
  "nearest neighbor", "HNSW", "graph traversal", "knowledge graph", "PageRank",
  "Node2Vec", "embedding search", "similarity search in SQLite".
license: MIT
compatibility: >
  Requires muninn extension. Python: `pip install sqlite-muninn`.
  Node.js: `npm install sqlite-muninn`. C: download amalgamation from GitHub Releases.
metadata:
  author: joshpeak
  version: "{{VERSION}}"
  repository: https://github.com/user/sqlite-muninn
allowed-tools: Bash(sqlite3:*), Bash(python:*), Bash(node:*)
---

# muninn — HNSW Vector Search + Graph Traversal for SQLite

Zero-dependency C11 SQLite extension. Three subsystems in one `.load`:
HNSW approximate nearest neighbor search, graph traversal TVFs, and Node2Vec.

## Installation Check

!`python -c "import sqlite_muninn; print(f'sqlite-muninn {sqlite_muninn.__version__} installed')" 2>/dev/null || echo "Not installed. Run: pip install sqlite-muninn"`

## Quick Start (Python)

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)
```

## Quick Start (Node.js)

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);
```

## Quick Start (SQLite CLI)

```sql
.load ./muninn
```

## HNSW Vector Index

### Create an index

```sql
CREATE VIRTUAL TABLE my_embeddings USING hnsw_index(
    id INTEGER PRIMARY KEY,
    embedding FLOAT32[384],       -- dimension MUST match your vectors
    metric_type TEXT DEFAULT 'L2'  -- L2 | cosine | inner_product
);
```

### Insert vectors

Vectors are raw float32 blobs (little-endian). In Python:
```python
import struct
dim = 384
vector = [0.1, 0.2, ...]  # len == dim
blob = struct.pack(f'{dim}f', *vector)
db.execute("INSERT INTO my_embeddings(id, embedding) VALUES (?, ?)", (1, blob))
```

In Node.js:
```javascript
const dim = 384;
const vector = new Float32Array([0.1, 0.2, /* ... */]);  // length == dim
const blob = Buffer.from(vector.buffer);
db.prepare("INSERT INTO my_embeddings(id, embedding) VALUES (?, ?)").run(1, blob);
```

### Search (k-nearest neighbors)

```sql
SELECT id, distance
FROM my_embeddings
WHERE embedding MATCH ?    -- query vector as float32 blob
  AND k = 10;             -- return top-10 nearest
```

## Graph Traversal TVFs

Graph TVFs operate on ANY existing SQLite table with source/target columns.
They do NOT require HNSW — they work on plain relational edge tables.

### BFS traversal

```sql
SELECT * FROM bfs(
    'edges',        -- table name
    'source',       -- source column
    'target',       -- target column
    1               -- start node ID
);
```

### Shortest path

```sql
SELECT * FROM shortest_path(
    'edges', 'source', 'target',
    1,              -- from node
    42              -- to node
);
```

### Available TVFs

| Function | Purpose |
|----------|---------|
| `bfs(table, src_col, dst_col, start)` | Breadth-first traversal |
| `dfs(table, src_col, dst_col, start)` | Depth-first traversal |
| `shortest_path(table, src_col, dst_col, from, to)` | Shortest path between two nodes |
| `connected_components(table, src_col, dst_col)` | Find all connected components |
| `pagerank(table, src_col, dst_col)` | Compute PageRank scores |

## Node2Vec Embedding Generation

Generate graph embeddings and store them directly in an HNSW index:

```sql
-- Create an HNSW index to receive the embeddings
CREATE VIRTUAL TABLE node_embeddings USING hnsw_index(
    id INTEGER PRIMARY KEY,
    embedding FLOAT32[64]
);

-- Train Node2Vec on an edge table, write embeddings into the HNSW index
SELECT node2vec_train(
    'edges',            -- edge table
    'source',           -- source column
    'target',           -- target column
    'node_embeddings',  -- destination HNSW table
    64                  -- embedding dimension
);
```

## Common Mistakes

- **DO NOT** pass vectors as JSON arrays — they must be raw float32 blobs
- **DO NOT** forget `db.enable_load_extension(True)` before loading (Python)
- **DO NOT** mismatch vector dimensions — insert dim must match CREATE TABLE dim
- **DO NOT** use on macOS system Python without Homebrew Python or pysqlite3-binary
  (Apple's SQLite has `SQLITE_OMIT_LOAD_EXTENSION`)
- **DO** call `enable_load_extension(False)` after loading for security
- **DO** use `struct.pack(f'{dim}f', *values)` for vector encoding in Python
- **DO** use `Buffer.from(new Float32Array(values).buffer)` for vectors in Node.js

## Platform Notes

- macOS: System Python's sqlite3 disables load_extension(). Use Homebrew Python
  (`brew install python`) or install `pysqlite3-binary`.
- All platforms: The extension is a native binary (.so/.dylib/.dll) — it must match
  your OS and architecture.

## Further Reading

See `references/` for detailed per-language cookbooks:
- [cookbook-python.md](references/cookbook-python.md) — Semantic search, RAG, batch loading
- [cookbook-node.md](references/cookbook-node.md) — Express endpoints, Bun, buffer helpers
- [cookbook-c.md](references/cookbook-c.md) — Static linking, sqlite3_auto_extension
- [cookbook-sql.md](references/cookbook-sql.md) — Pure SQL workflows, graph pipelines
- [vector-encoding.md](references/vector-encoding.md) — Cross-language float32 blob format
- [platform-caveats.md](references/platform-caveats.md) — macOS, Windows, glibc details
````

### Claude Code Plugin Manifest

Following the QMD `.claude-plugin/marketplace.json` pattern, we register muninn as a Claude Code plugin:

```json
{
  "name": "muninn",
  "owner": {
    "name": "joshpeak",
    "email": "josh@example.com"
  },
  "plugins": [
    {
      "name": "muninn",
      "source": "./",
      "description": "HNSW vector search, graph traversal, and Node2Vec for SQLite.",
      "version": "0.1.0",
      "author": { "name": "joshpeak" },
      "repository": "https://github.com/user/sqlite-muninn",
      "license": "MIT",
      "keywords": ["sqlite", "vector", "hnsw", "graph", "node2vec", "search"],
      "skills": ["./skills/"]
    }
  ]
}
```

This enables Claude Code to discover the skill when the plugin is installed, making the full `SKILL.md` and all `references/` files available as context when a user's task matches the trigger phrases.

### Per-Language Reference Cookbooks

The `references/` directory contains deeper patterns for each language ecosystem. These are the recipes an AI coding tool needs to solve real tasks — not just toy examples, but production patterns with error handling and performance considerations.

#### `references/cookbook-python.md`

| Pattern | What the AI learns |
|---------|-------------------|
| **Semantic search over documents** | Embed text with sentence-transformers, `struct.pack` vectors, store in HNSW, search by query vector |
| **Knowledge graph + vector hybrid** | Store entities as nodes with embeddings, query by both graph traversal and vector similarity |
| **Node2Vec → clustering** | Generate graph embeddings, extract as numpy arrays, feed to scikit-learn |
| **Batch vector insert** | Efficient bulk loading with `executemany()`, pre-packed struct buffers |
| **RAG retrieval** | Retrieve top-k similar chunks, expand context with BFS over citation graph |
| **Incremental index updates** | Insert/delete/update vectors in a live HNSW index |

#### `references/cookbook-node.md`

| Pattern | What the AI learns |
|---------|-------------------|
| **Express.js search endpoint** | API route: receive query → embed → HNSW search → JSON response |
| **Bun SQLite integration** | `bun:sqlite` with `muninn` for edge-native search |
| **`Float32Array` ↔ `Buffer`** | Correct vector encoding/decoding for insert and search |
| **better-sqlite3 transactions** | Batch inserts with `db.transaction()` for performance |

#### `references/cookbook-c.md`

| Pattern | What the AI learns |
|---------|-------------------|
| **Static linking** | Compile muninn into your application with `sqlite3_auto_extension()` |
| **Loadable extension** | `sqlite3_load_extension()` with proper error handling |
| **Vector preparation** | Allocating `float[]` buffers, binding as `SQLITE_BLOB` with `sqlite3_bind_blob()` |
| **Thread safety** | One `sqlite3*` connection per thread; HNSW index is connection-scoped |

#### `references/cookbook-sql.md`

| Pattern | What the AI learns |
|---------|-------------------|
| **Interactive exploration** | `.load`, create index, insert sample vectors, search — a full CLI session |
| **Graph analysis pipeline** | Create edge table → PageRank → Node2Vec → similarity search — end to end |
| **Hybrid query** | Combine HNSW vector search with graph BFS in a single workflow |

#### `references/vector-encoding.md`

Cross-language reference for the float32 blob format — the single biggest source of errors when AI tools generate muninn code:

| Language | Encode | Decode |
|----------|--------|--------|
| Python | `struct.pack(f'{dim}f', *values)` | `struct.unpack(f'{dim}f', blob)` |
| Node.js | `Buffer.from(new Float32Array(values).buffer)` | `new Float32Array(buffer.buffer, buffer.byteOffset, dim)` |
| C | `float vec[dim]; sqlite3_bind_blob(stmt, col, vec, dim*sizeof(float), SQLITE_TRANSIENT);` | `const float *vec = sqlite3_column_blob(stmt, col);` |
| Rust | `bytemuck::cast_slice::<f32, u8>(&vec)` | `bytemuck::cast_slice::<u8, f32>(blob)` |
| Go | `math.Float32bits()` → `binary.LittleEndian.PutUint32()` | `binary.LittleEndian.Uint32()` → `math.Float32frombits()` |

### Distribution Integration

The `skills/` directory and its contents must ship inside every package, not just live in the repo:

#### Python (PyPI)

```
sqlite_muninn/
    __init__.py
    muninn.so              # platform binary
    skills/
        muninn/
            SKILL.md
            references/
                cookbook-python.md
                vector-encoding.md
                platform-caveats.md
```

Include in `pyproject.toml`:
```toml
[tool.setuptools.package-data]
sqlite_muninn = ["skills/**/*.md"]
```

#### Node.js (NPM)

```
sqlite-muninn/
    index.mjs
    index.cjs
    index.d.ts
    skills/
        muninn/
            SKILL.md
            references/
                cookbook-node.md
                vector-encoding.md
                platform-caveats.md
```

Include in `package.json`:
```json
{
  "files": ["index.mjs", "index.cjs", "index.d.ts", "skills/**/*.md"]
}
```

#### C/C++ (Amalgamation Tarball)

```
muninn-amalgamation/
    muninn.c
    muninn.h
    skills/
        muninn/
            SKILL.md
            references/
                cookbook-c.md
                cookbook-sql.md
                vector-encoding.md
                platform-caveats.md
```

#### GitHub Repo (Source)

The repo is the source of truth. All files live at `skills/muninn/` at the repo root. Each language package includes the full `SKILL.md` plus only the relevant reference files for that ecosystem.

### Cross-Tool Compatibility

Different AI tools discover instructions differently. The `SKILL.md` with YAML frontmatter is the canonical format; tool-specific files can be generated from it:

| Tool | Discovery Mechanism | How muninn Gets Found |
|------|---------------------|--------------------------|
| **Claude Code** | `.claude-plugin/` + `skills/` | Native plugin discovery; `SKILL.md` frontmatter parsed directly |
| **Claude Code (manual)** | `CLAUDE.md` in consumer project | `python -m sqlite_muninn init --claude` generates snippet |
| **Cursor** | `.cursorrules` in consumer project | `python -m sqlite_muninn init --cursor` generates rules |
| **GitHub Copilot** | README + adjacent files | README links to `skills/muninn/SKILL.md` |
| **Aider** | `.aider.conf.yml` conventions | Link to SKILL.md in conventions file |
| **Windsurf** | `.windsurfrules` | `python -m sqlite_muninn init --windsurf` generates rules |
| **Context7** | Indexed documentation | Published docs include SKILL.md patterns |
| **Any tool** | Reads files adjacent to imported package | `skills/` directory discoverable next to `__init__.py` or `index.mjs` |

**`sqlite_muninn init` CLI helper** (future):

```bash
# Generate tool-specific instruction files from canonical SKILL.md
python -m sqlite_muninn init --claude    # → prints CLAUDE.md snippet to stdout
python -m sqlite_muninn init --cursor    # → prints .cursorrules content
python -m sqlite_muninn init --windsurf  # → prints .windsurfrules content
python -m sqlite_muninn init --all       # → writes all tool-specific files
```

### Maintenance Strategy

The skill files are **release artifacts** — versioned and validated alongside the code:

1. **Source of truth:** `skills/muninn/` at the repo root (human-editable, reviewed in PRs)
2. **Build step:** `make skill` copies the skills directory into each package, stamping the version in YAML frontmatter
3. **CI validation:** A job extracts and executes every code block from `SKILL.md` and all `references/*.md` files against the built extension
4. **Versioned with releases:** The `SKILL.md` in version 0.2.0's PyPI package describes 0.2.0's API, not trunk

```makefile
skill: skills/muninn/SKILL.md                ## Stamp version into skill files
	@VERSION=$$(cat VERSION); \
	for f in skills/muninn/SKILL.md skills/muninn/references/*.md; do \
	    sed "s/{{VERSION}}/$$VERSION/g" "$$f" > "dist/$${f}"; \
	done
```

### CI Validation: Executable Documentation

The most dangerous failure mode is a `SKILL.md` that contains wrong SQL syntax. AI tools will faithfully reproduce the wrong syntax, and developers will file bugs saying "the AI told me to do X and it didn't work."

**Prevention:** Extract and execute every fenced code block from `SKILL.md` and all reference files:

```yaml
validate-skill-md:
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
    - run: make all
    - name: Validate skill examples
      run: python scripts/validate_skill_examples.py skills/muninn/
```

The validation script:
1. Recursively finds all `.md` files under `skills/muninn/`
2. Parses fenced code blocks (```sql, ```python, ```javascript)
3. Runs SQL blocks against a fresh SQLite connection with muninn loaded
4. Runs Python blocks in a subprocess with muninn importable
5. Reports any blocks that error out (with file path and line number)

This ensures that every code example across the entire skill directory **actually works** against the current build. When a contributor changes the SQL interface, CI will catch the stale skill docs before they ship.

### Why This Is a Competitive Advantage

Most SQLite extensions (and most libraries in general) rely on:
- README files that AI tools may or may not find
- Documentation sites that require web fetching
- Stack Overflow answers that may be outdated

By shipping a structured `skills/` directory **inside every package**, `muninn` becomes the path of least resistance for any AI coding tool. When a developer says "add vector search to my SQLite app," the AI tool that can find and read `muninn`'s `SKILL.md` will produce correct, working code on the first attempt — while competitors require multiple rounds of debugging hallucinated syntax.

This is especially powerful for a niche category (SQLite extensions) where AI training data is sparse. The `SKILL.md` essentially **injects correct knowledge** at the point of use, bypassing the training data gap entirely.

The QMD project demonstrates this works in practice — its skill file makes `qmd` trivially usable by Claude Code despite being a relatively new, niche CLI tool. We're applying the same pattern to a SQLite extension, where the knowledge gap is even wider.

---

## Recommended Implementation Order

### Phase 1: CI Foundation (No Publishing)

_Get green builds on all platforms before thinking about distribution._

1. **Vendor SQLite headers** — `scripts/vendor.sh` to download amalgamation
2. **Improve SQLite detection** — `pkg-config` → Homebrew → manual override chain
3. **Add Windows build support** — MSVC build command in CI (or `Makefile.msc`)
4. **Create `.github/workflows/ci.yml`** — Build + test on 4 runners (Linux x64, Linux arm64, macOS universal, Windows)
5. **Add sanitizer jobs** — ASan+UBSan, MSan on Linux
6. **Add SQLite version matrix** — Test against 3.21, 3.38, 3.44, latest

### Phase 2: C Distribution (Amalgamation + Install)

_Make muninn consumable by C/C++ projects._

7. **Create `scripts/amalgamate.sh`** — Produces `dist/muninn.c` + `dist/muninn.h`
8. **Add `make amalgamation` target** — Generates the amalgamation
9. **Add `VERSION` file** — Single source of truth for all packages
10. **Add `muninn.pc.in`** — pkg-config template
11. **Add `make install` / `make uninstall`** — Standard `PREFIX`/`DESTDIR` conventions
12. **Add `CMakeLists.txt`** — Support FetchContent + find_package for CMake consumers
13. **Update release workflow** — Upload amalgamation tarball to GitHub Releases

### Phase 3: Python Distribution

_Ship `pip install sqlite-muninn`._

14. **Create `bindings/python/`** — `__init__.py` with `loadable_path()` + `load()`
15. **Create `scripts/build_wheels.py`** — Assembles platform-tagged wheels from CI artifacts
16. **Add PyPI trusted publisher** — Configure at pypi.org
17. **Add `publish-pypi` job** to `release.yml`
18. **Test the full flow** — Tag → Release → PyPI

### Phase 4: Node.js Distribution

_Ship `npm install sqlite-muninn` (native) and `npm install sqlite-muninn-wasm` (browser)._

19. **Create `npm/` directory structure** — Main package + 5 platform packages
20. **Write `index.mjs`, `index.cjs`, `index.d.ts`** — Wrapper API
21. **Add npm trusted publisher** — Configure per-package at npmjs.com
22. **Add `publish-npm` job** to `release.yml`
23. **Test with `better-sqlite3` and `node:sqlite`**

### Phase 5: WASM Build

_Ship `sqlite-muninn-wasm` for browser and edge runtimes._

24. **Create `src/sqlite3_wasm_extra_init.c`** — Static extension registration
25. **Create `scripts/build_wasm.sh`** — Emscripten build script (SQLite amalgamation + muninn amalgamation)
26. **Add `build-wasm` job** to release workflow — `mymindstorm/setup-emsdk` action
27. **Investigate WASM SIMD** (`-msimd128`) — Recover `vec_math.c` performance where possible
28. **Create `npm/sqlite-muninn-wasm/`** — Platform-independent npm package
29. **Add `publish-wasm` job** to `release.yml`
30. **Test in browser** — Verify HNSW + graph TVFs work end-to-end in WASM

### Phase 6: Agent-Ready Documentation (SKILL.md + Skills Directory)

_Make muninn the easiest SQLite extension for AI coding tools to use correctly. Follows the [QMD skills pattern](https://github.com/tobi/qmd)._

31. **Create `skills/muninn/SKILL.md`** — YAML frontmatter (name, description, triggers, allowed-tools, compatibility) + structured usage guide with Quick Starts, SQL reference, and Common Mistakes
32. **Create `skills/muninn/references/`** — Per-language cookbooks: `cookbook-python.md`, `cookbook-node.md`, `cookbook-c.md`, `cookbook-sql.md`, plus cross-cutting `vector-encoding.md` and `platform-caveats.md`
33. **Create `.claude-plugin/marketplace.json`** — Claude Code plugin manifest pointing to `skills/` directory
34. **Add `make skill` target** — Version-stamp `{{VERSION}}` in all skill markdown files during build
35. **Write `scripts/validate_skill_examples.py`** — Recursively extract and execute all fenced code blocks from `skills/muninn/**/*.md` against the built extension
36. **Add `validate-skill-md` CI job** — Ensure agent documentation never drifts from the actual API
37. **Include `skills/` in package manifests** — Add to `pyproject.toml` package-data, `package.json` files array, and amalgamation tarball (each ecosystem gets SKILL.md + relevant reference files only)
38. **Build `python -m sqlite_muninn init` CLI** — Generate tool-specific instruction files (CLAUDE.md, .cursorrules, .windsurfrules) from canonical SKILL.md
39. **Submit to Context7** — Ensure SKILL.md patterns are indexed for tools that use Context7 documentation lookup

### Phase 7: Ecosystem (Optional)

40. **Homebrew formula** — Custom tap for `brew install sqlite-muninn`
41. **Fuzz testing** — libFuzzer harnesses for `hnsw_algo` and `graph_tvf`
42. **Performance regression tracking** — Bencher or github-action-benchmark
43. **Bundled SQLite package** — `sqlite-muninn-bundled` with baked-in extension
44. **vcpkg / Conan ports** — When demand warrants

---

## Appendix: Prior Art & References

### Projects Studied

| Project | PyPI | NPM | Approach |
|---------|------|-----|----------|
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | `py3-none-{platform}` wheels | Platform optionalDeps | Gold standard; uses `sqlite-dist` |
| [sqlean.py](https://github.com/nalgeon/sqlean.py) | CPython replacement module | N/A | Bundles custom SQLite with 12 extensions |
| [pysqlite3](https://github.com/coleifer/pysqlite3) | CPython extension + amalgamation | N/A | Custom SQLite build |
| [better-sqlite3](https://github.com/WiseLibs/better-sqlite3) | N/A | N-API + prebuild-install | Bundles SQLite amalgamation |
| [sql.js](https://github.com/sql-js/sql.js) | N/A | WASM (Emscripten) | SQLite in WebAssembly; no dynamic extensions |
| [esbuild](https://github.com/evanw/esbuild) | N/A | Platform optionalDeps | Pioneered the pattern at scale |
| [QMD](https://github.com/tobi/qmd) | N/A | N/A | Pioneered `skills/` directory + `SKILL.md` + `.claude-plugin/` pattern for AI-native distribution |

### Key Tools

| Tool | Purpose |
|------|---------|
| [sqlite-dist](https://github.com/asg017/sqlite-dist) | Multi-ecosystem packaging (WIP but production-used) |
| [cibuildwheel](https://github.com/pypa/cibuildwheel) | Cross-platform Python wheel building (for CPython extensions) |
| [actions/setup-sqlite@v1](https://github.com/marketplace/actions/setup-sqlite-environment) | SQLite version management in GitHub Actions |
| [prebuildify](https://github.com/prebuild/prebuildify) | Bundle N-API prebuilds in npm packages |
| [git-cliff](https://github.com/orhun/git-cliff) | Changelog generation from Conventional Commits |
| [Bencher](https://bencher.dev/) | Continuous benchmarking / performance regression tracking |

### Key Documentation

- [SQLite Loadable Extensions](https://sqlite.org/loadext.html)
- [SQLite Virtual Table Mechanism](https://sqlite.org/vtab.html)
- [SQLite Compile-Time Options](https://sqlite.org/compile.html)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [npm Trusted Publishing](https://docs.npmjs.com/trusted-publishers/)
- [Python Wheel Platform Tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)
