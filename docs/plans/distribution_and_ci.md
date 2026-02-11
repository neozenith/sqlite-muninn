# Distribution & CI/CD Plan

> **Status:** Research / Proposal
> **Date:** 2026-02-11
> **Scope:** Publishing `vec_graph` to PyPI and NPM with cross-platform CI/CD

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
10. [Recommended Implementation Order](#recommended-implementation-order)

---

## Executive Summary

The goal is to make `vec_graph` installable via `pip install vec-graph` and `npm install vec-graph`, with precompiled binaries for all major platforms. The approach follows the pattern established by **sqlite-vec** (Alex Garcia), which is the gold standard for SQLite extension distribution.

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

`vec-graph` on PyPI → `import vec_graph` in Python.

### Package Structure

```
vec_graph/
    __init__.py      # loadable_path() + load() + version
    vec_graph.so     # Linux (or .dylib on macOS, .dll on Windows)
```

The binary sits alongside `__init__.py`. No sub-packages needed.

### API Surface

```python
# vec_graph/__init__.py
import os
import sqlite3

__version__ = "0.1.0"
__version_info__ = tuple(__version__.split("."))

def loadable_path() -> str:
    """Return path to the vec_graph loadable extension (without file extension).

    SQLite's load_extension() automatically appends .so/.dylib/.dll.
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "vec_graph"))

def load(conn: sqlite3.Connection) -> None:
    """Load vec_graph into the given SQLite connection."""
    conn.load_extension(loadable_path())
```

User-facing usage:

```python
import sqlite3
import vec_graph

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
vec_graph.load(db)
db.enable_load_extension(False)

# Now HNSW, graph TVFs, and node2vec are available
```

### Wheel Tags

The key insight: since `vec_graph` is loaded via `sqlite3.load_extension()` (not Python's C extension mechanism), it does **not** link against `libpython`. The wheel tag is:

```
vec_graph-0.1.0-py3-none-{platform}.whl
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

1. Configure at `https://pypi.org/manage/project/vec-graph/settings/publishing/`
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

`vec-graph` on NPM (main wrapper) + `@vec-graph/{platform}` platform packages.

### Architecture: Platform-Specific Optional Dependencies

This is the **esbuild pattern**, also used by sqlite-vec, Prisma, and SWC:

```
npm/
  vec-graph/                    # Main wrapper package
    package.json                # optionalDependencies → platform packages
    index.mjs                   # ESM entry
    index.cjs                   # CJS entry
    index.d.ts                  # TypeScript declarations
  @vec-graph/
    darwin-arm64/               # macOS Apple Silicon
      package.json              # os: ["darwin"], cpu: ["arm64"]
      vec_graph.dylib
    darwin-x64/                 # macOS Intel
      package.json
      vec_graph.dylib
    linux-x64/                  # Linux x86_64
      package.json
      vec_graph.so
    linux-arm64/                # Linux ARM64
      package.json
      vec_graph.so
    win32-x64/                  # Windows x86_64
      package.json
      vec_graph.dll
```

### Platform Package Structure

Each platform package is minimal:

```json
{
  "name": "@vec-graph/darwin-arm64",
  "version": "0.1.0",
  "os": ["darwin"],
  "cpu": ["arm64"],
  "files": ["vec_graph.dylib"]
}
```

npm/yarn/pnpm automatically installs **only** the matching platform package.

### Main Package

```json
{
  "name": "vec-graph",
  "version": "0.1.0",
  "main": "index.cjs",
  "module": "index.mjs",
  "types": "index.d.ts",
  "optionalDependencies": {
    "@vec-graph/darwin-arm64": "0.1.0",
    "@vec-graph/darwin-x64": "0.1.0",
    "@vec-graph/linux-x64": "0.1.0",
    "@vec-graph/linux-arm64": "0.1.0",
    "@vec-graph/win32-x64": "0.1.0"
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
  "darwin-arm64": { pkg: "@vec-graph/darwin-arm64", ext: "dylib" },
  "darwin-x64":   { pkg: "@vec-graph/darwin-x64",   ext: "dylib" },
  "linux-x64":    { pkg: "@vec-graph/linux-x64",    ext: "so" },
  "linux-arm64":  { pkg: "@vec-graph/linux-arm64",   ext: "so" },
  "win32-x64":    { pkg: "@vec-graph/win32-x64",    ext: "dll" },
};

export function getLoadablePath() {
  const key = `${platform}-${arch}`;
  const target = PLATFORM_MAP[key];
  if (!target) {
    throw new Error(`Unsupported platform: ${key}. Supported: ${Object.keys(PLATFORM_MAP).join(", ")}`);
  }
  const loadablePath = join(__dirname, "..", target.pkg, `vec_graph.${target.ext}`);
  if (!statSync(loadablePath, { throwIfNoEntry: false })) {
    throw new Error(
      `vec_graph binary not found for ${key}. ` +
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
| Users need specific SQLite version or flags | Offer a `vec-graph-sqlite` package with bundled SQLite |
| Browser/WASM usage | Requires statically linked SQLite+extension build |

### Bundled SQLite Package (Optional, Future)

If demand warrants it, create a `vec-graph-sqlite` package that bundles the SQLite amalgamation with `vec_graph` compiled in:

```python
# This would be a CPython extension (like pysqlite3/sqlean.py)
# Replaces the built-in sqlite3 module
import vec_graph_sqlite as sqlite3  # drop-in replacement with vec_graph baked in
```

**Implementation approach (from sqlean.py):**

1. Download SQLite amalgamation (`sqlite3.c` + `sqlite3.h`)
2. Compile with `vec_graph` statically linked via `SQLITE_EXTRA_INIT` / `sqlite3_auto_extension()`
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

Compile SQLite + `vec_graph` to WebAssembly via Emscripten for browser and edge runtime usage. Extensions cannot be dynamically loaded in WASM — they must be statically linked at compile time.

**Static registration entry point:**

```c
// src/sqlite3_wasm_extra_init.c
#include "sqlite3.h"
extern int sqlite3_vecgraph_init(sqlite3*, char**, const sqlite3_api_routines*);

int sqlite3_wasm_extra_init(const char *z) {
    return sqlite3_auto_extension((void(*)(void))sqlite3_vecgraph_init);
}
```

**Build process (Emscripten):**

```bash
# 1. Download SQLite amalgamation
wget https://www.sqlite.org/2025/sqlite-amalgamation-3510000.zip
unzip sqlite-amalgamation-3510000.zip

# 2. Compile SQLite + vec_graph → WASM
emcc -O2 -s WASM=1 -s EXPORTED_FUNCTIONS='["_sqlite3_open", ...]' \
     -DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_JSON1 \
     sqlite-amalgamation-3510000/sqlite3.c \
     dist/vec_graph.c \
     src/sqlite3_wasm_extra_init.c \
     -o dist/vec_graph_sqlite3.js
```

**NPM distribution:** Ship as a separate `vec-graph-wasm` package (not bundled with the native packages). The WASM binary is platform-independent — a single package works everywhere:

```json
{
  "name": "vec-graph-wasm",
  "version": "0.1.0",
  "files": ["vec_graph_sqlite3.js", "vec_graph_sqlite3.wasm"],
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
      with: { name: wasm, path: "dist/vec_graph_sqlite3.*" }
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
   -Isrc -o vec_graph_arm64.dylib src/*.c -lm

cc -arch x86_64 -O2 -std=c11 -fPIC -dynamiclib -undefined dynamic_lookup \
   -Isrc -o vec_graph_x86_64.dylib src/*.c -lm

# Combine into a single universal binary
lipo -create vec_graph_arm64.dylib vec_graph_x86_64.dylib -output vec_graph.dylib

# Verify both slices are present
lipo -info vec_graph.dylib
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
          src\vec_graph.c src\hnsw_vtab.c src\hnsw_algo.c ^
          src\graph_tvf.c src\node2vec.c src\vec_math.c ^
          src\priority_queue.c src\id_validate.c ^
          /Fe:vec_graph.dll
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
objdump -T vec_graph.so | grep GLIBC_ | sed 's/.*GLIBC_//' | sort -Vu
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
        with: { name: linux-x86_64, path: vec_graph.so }

  build-linux-arm64:
    runs-on: ubuntu-22.04-arm
    steps:
      - uses: actions/checkout@v4
      - run: make all
      - uses: actions/upload-artifact@v4
        with: { name: linux-arm64, path: vec_graph.so }

  build-macos-universal:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - run: make ARCH=arm64 all
      - run: mv vec_graph.dylib vec_graph_arm64.dylib
      - run: make clean && make ARCH=x86_64 all
      - run: mv vec_graph.dylib vec_graph_x86_64.dylib
      - run: lipo -create vec_graph_arm64.dylib vec_graph_x86_64.dylib -output vec_graph.dylib
      - uses: actions/upload-artifact@v4
        with: { name: macos-universal, path: vec_graph.dylib }

  build-windows:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - uses: ilammy/msvc-dev-cmd@v1
      - shell: cmd
        run: cl.exe /O2 /MT /W4 /LD /Isrc src\*.c /Fe:vec_graph.dll
      - uses: actions/upload-artifact@v4
        with: { name: windows-x86_64, path: vec_graph.dll }
```

This uses **4 CI jobs** to produce **5 platform binaries** (the macOS job produces a universal binary containing both architectures).

---

## C/C++ Dependency Distribution

How should C/C++ consumers depend on `vec_graph`? Unlike Python/Node.js, the C ecosystem has no single dominant package manager. Instead, there are several parallel distribution channels, ordered here by priority for the SQLite extension ecosystem.

### Priority 1: Amalgamation (Single `.c` + `.h` File)

This is the **primary distribution format** for the C ecosystem — it's how SQLite itself, sqlite-vec, and most SQLite extensions are consumed.

**What it is:** All source files concatenated into a single `vec_graph.c`, plus a public API header `vec_graph.h`. Consumer just adds two files to their project.

**Consumer usage:**

```bash
# Download from GitHub release
wget https://github.com/user/sqlite-vector-graph/releases/download/v0.1.0/vec_graph-amalgamation.tar.gz
tar xf vec_graph-amalgamation.tar.gz

# Build as loadable extension
gcc -O2 -fPIC -shared vec_graph.c -o vec_graph.so -lm           # Linux
cc -O2 -fPIC -dynamiclib vec_graph.c -o vec_graph.dylib -lm     # macOS

# Or compile into their application with static linking
gcc -O2 myapp.c vec_graph.c -lsqlite3 -lm -o myapp
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
OUT=dist/vec_graph.c

mkdir -p dist
cat > "$OUT" <<HEADER
/* vec_graph amalgamation - v${VERSION}
 * Generated $(date -u +%Y-%m-%d)
 * https://github.com/user/sqlite-vector-graph
 */
HEADER

# Concatenate headers (removing internal #include "..." directives)
for f in src/vec_math.h src/priority_queue.h src/hnsw_algo.h \
         src/id_validate.h src/hnsw_vtab.h src/graph_tvf.h \
         src/node2vec.h src/vec_graph.h; do
    echo "/* ---- $f ---- */"
    grep -v '#include "' "$f"
done >> "$OUT"

# Concatenate implementations
for f in src/vec_math.c src/priority_queue.c src/hnsw_algo.c \
         src/id_validate.c src/hnsw_vtab.c src/graph_tvf.c \
         src/node2vec.c src/vec_graph.c; do
    echo "/* ---- $f ---- */"
    grep -v '#include "' "$f"
done >> "$OUT"

# Copy public header
cp src/vec_graph.h dist/

echo "Amalgamation: dist/vec_graph.c ($(wc -l < "$OUT") lines)"
```

Option B — Use [cwoffenden/combiner](https://github.com/cwoffenden/combiner) for recursive `#include` resolution:

```bash
python combiner/combine.py -r src -o dist/vec_graph.c src/vec_graph.c
```

**Makefile target:**

```makefile
amalgamation: dist/vec_graph.c dist/vec_graph.h   ## Create amalgamation

dist/vec_graph.c dist/vec_graph.h: $(SRC) $(wildcard src/*.h)
	bash scripts/amalgamate.sh
```

### Priority 2: Pre-compiled Binaries (GitHub Releases)

Ship `.so`/`.dylib`/`.dll` files as GitHub Release assets. Consumers download and use `sqlite3_load_extension()` directly. This is what the release workflow already produces.

### Priority 3: `make install` with pkg-config

The standard Unix library installation convention. Enables `pkg-config --cflags --libs vec_graph` for downstream Makefiles.

**`vec_graph.pc.in` template:**

```
prefix=@PREFIX@
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: vec_graph
Description: HNSW + Graph Traversal + Node2Vec SQLite Extension
Version: @VERSION@
Requires: sqlite3
Libs: -L${libdir} -lvec_graph -lm
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

vec_graph.pc: vec_graph.pc.in
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(VERSION)|g' $< > $@

install: vec_graph$(EXT) vec_graph.pc                ## Install library, headers, pkg-config
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -d $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 vec_graph$(EXT) $(DESTDIR)$(LIBDIR)/
	install -m 644 src/vec_graph.h $(DESTDIR)$(INCLUDEDIR)/
	install -m 644 vec_graph.pc $(DESTDIR)$(PKGCONFIGDIR)/

uninstall:                                           ## Remove installed files
	rm -f $(DESTDIR)$(LIBDIR)/vec_graph$(EXT)
	rm -f $(DESTDIR)$(INCLUDEDIR)/vec_graph.h
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/vec_graph.pc
```

**Consumer usage:**

```bash
# Install
make install PREFIX=/usr/local

# Consume from another Makefile
CFLAGS += $(shell pkg-config --cflags vec_graph)
LDFLAGS += $(shell pkg-config --libs vec_graph)
```

### Priority 4: CMake FetchContent

For CMake-based consumers, provide a `CMakeLists.txt` that supports both `FetchContent` (build from source) and `find_package` (use installed version).

**`CMakeLists.txt` at repo root:**

```cmake
cmake_minimum_required(VERSION 3.14)
project(vec_graph VERSION 0.1.0 LANGUAGES C)

find_package(SQLite3 REQUIRED)

add_library(vec_graph
    src/vec_graph.c src/hnsw_vtab.c src/hnsw_algo.c
    src/graph_tvf.c src/node2vec.c src/vec_math.c
    src/priority_queue.c src/id_validate.c
)

target_include_directories(vec_graph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(vec_graph PUBLIC SQLite::SQLite3 PRIVATE m)
target_compile_features(vec_graph PUBLIC c_std_11)
set_target_properties(vec_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Only build tests when this is the top-level project (not when consumed via FetchContent)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    enable_testing()
    add_subdirectory(test)
endif()

# Install rules (for find_package)
include(GNUInstallDirs)
install(TARGETS vec_graph EXPORT vec_graph-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES src/vec_graph.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vec_graph-targets
    FILE vec_graph-config.cmake
    NAMESPACE vec_graph::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vec_graph
)
```

**Consumer CMakeLists.txt:**

```cmake
include(FetchContent)
FetchContent_Declare(
    vec_graph
    GIT_REPOSITORY https://github.com/user/sqlite-vector-graph.git
    GIT_TAG        v0.1.0       # Pin to a release tag
)
FetchContent_MakeAvailable(vec_graph)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE vec_graph)
```

The `CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR` guard is critical — it prevents tests and dev targets from building when consumed as a dependency.

### Priority 5: Git Submodules

The oldest and simplest approach. Consumer adds the repo as a submodule:

```bash
git submodule add -b v0.1.0 https://github.com/user/sqlite-vector-graph.git vendor/vec_graph
```

Then in their Makefile:

```makefile
CFLAGS += -Ivendor/vec_graph/src
SRCS   += vendor/vec_graph/src/vec_graph.c vendor/vec_graph/src/hnsw_vtab.c ...
```

No changes needed to `vec_graph` — submodules just clone the repo at a pinned commit. But submodules are considered legacy; FetchContent or the amalgamation is preferred.

### Priority 6: Homebrew Formula (macOS)

```ruby
class VecGraph < Formula
  desc "HNSW + graph traversal + Node2Vec SQLite extension"
  homepage "https://github.com/user/sqlite-vector-graph"
  url "https://github.com/user/sqlite-vector-graph/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<sha256>"
  license "MIT"
  depends_on "sqlite"

  def install
    system "make", "all", "SQLITE_PREFIX=#{Formula["sqlite"].opt_prefix}"
    lib.install "vec_graph.dylib"
    include.install "src/vec_graph.h"
    (lib/"pkgconfig").install "vec_graph.pc"
  end

  test do
    system Formula["sqlite"].opt_bin/"sqlite3", ":memory:",
           ".load #{lib}/vec_graph", "SELECT 1"
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
| **Homebrew** | `brew install vec-graph` | Low (formula file) | macOS developers |
| **pip / npm** | `pip install` / `npm install` | Medium (covered above) | Python / JS developers |
| **vcpkg / Conan** | `vcpkg install vec-graph` | Medium | Windows / cross-platform |

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
          path: vec_graph.${{ matrix.ext }}

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
            npm publish npm/@vec-graph/$target --provenance --access public
          done
      - name: Publish main package
        run: npm publish npm/vec-graph --provenance --access public

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
            linux-x64/vec_graph.so
            linux-arm64/vec_graph.so
            darwin-arm64/vec_graph.dylib
            darwin-x64/vec_graph.dylib
            win32-x64/vec_graph.dll
```

### Build System Changes Needed

The current Makefile needs these additions for cross-platform CI:

1. **Windows MSVC support** — Add a `Makefile.msc` or `build_windows.bat`:
   ```bat
   cl /O2 /W3 /LD src\vec_graph.c src\hnsw_vtab.c src\hnsw_algo.c ^
      src\graph_tvf.c src\node2vec.c src\vec_math.c src\priority_queue.c ^
      src\id_validate.c /Fe:vec_graph.dll
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

_Make vec_graph consumable by C/C++ projects._

7. **Create `scripts/amalgamate.sh`** — Produces `dist/vec_graph.c` + `dist/vec_graph.h`
8. **Add `make amalgamation` target** — Generates the amalgamation
9. **Add `VERSION` file** — Single source of truth for all packages
10. **Add `vec_graph.pc.in`** — pkg-config template
11. **Add `make install` / `make uninstall`** — Standard `PREFIX`/`DESTDIR` conventions
12. **Add `CMakeLists.txt`** — Support FetchContent + find_package for CMake consumers
13. **Update release workflow** — Upload amalgamation tarball to GitHub Releases

### Phase 3: Python Distribution

_Ship `pip install vec-graph`._

14. **Create `bindings/python/`** — `__init__.py` with `loadable_path()` + `load()`
15. **Create `scripts/build_wheels.py`** — Assembles platform-tagged wheels from CI artifacts
16. **Add PyPI trusted publisher** — Configure at pypi.org
17. **Add `publish-pypi` job** to `release.yml`
18. **Test the full flow** — Tag → Release → PyPI

### Phase 4: Node.js Distribution

_Ship `npm install vec-graph` (native) and `npm install vec-graph-wasm` (browser)._

19. **Create `npm/` directory structure** — Main package + 5 platform packages
20. **Write `index.mjs`, `index.cjs`, `index.d.ts`** — Wrapper API
21. **Add npm trusted publisher** — Configure per-package at npmjs.com
22. **Add `publish-npm` job** to `release.yml`
23. **Test with `better-sqlite3` and `node:sqlite`**

### Phase 5: WASM Build

_Ship `vec-graph-wasm` for browser and edge runtimes._

24. **Create `src/sqlite3_wasm_extra_init.c`** — Static extension registration
25. **Create `scripts/build_wasm.sh`** — Emscripten build script (SQLite amalgamation + vec_graph amalgamation)
26. **Add `build-wasm` job** to release workflow — `mymindstorm/setup-emsdk` action
27. **Investigate WASM SIMD** (`-msimd128`) — Recover `vec_math.c` performance where possible
28. **Create `npm/vec-graph-wasm/`** — Platform-independent npm package
29. **Add `publish-wasm` job** to `release.yml`
30. **Test in browser** — Verify HNSW + graph TVFs work end-to-end in WASM

### Phase 6: Ecosystem (Optional)

31. **Homebrew formula** — Custom tap for `brew install vec-graph`
32. **Fuzz testing** — libFuzzer harnesses for `hnsw_algo` and `graph_tvf`
33. **Performance regression tracking** — Bencher or github-action-benchmark
34. **Bundled SQLite package** — `vec-graph-sqlite` with baked-in extension
35. **vcpkg / Conan ports** — When demand warrants

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
