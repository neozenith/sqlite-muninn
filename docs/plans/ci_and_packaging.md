# CI & Packaging Plan

> **Status:** Research / Proposal
> **Date:** 2026-02-12
> **Scope:** Build, test, and package `muninn` for all consumption modes — installable from git before any registry publishing
> **Supersedes:** CI portions of `distribution_and_ci.md`

---

## Table of Contents

1. [Goals & Consumption Modes](#goals--consumption-modes)
2. [Version Support Policy](#version-support-policy)
3. [Cross-Platform Build Matrix](#cross-platform-build-matrix)
4. [SQLite Header Vendoring](#sqlite-header-vendoring)
5. [Packaging: Python](#packaging-python)
6. [Packaging: Node.js / TypeScript](#packaging-nodejs--typescript)
7. [Packaging: WASM (Browser)](#packaging-wasm-browser)
8. [Packaging: C Library (Amalgamation)](#packaging-c-library-amalgamation)
9. [Packaging: SQLite CLI Extension](#packaging-sqlite-cli-extension)
10. [Testing Strategy](#testing-strategy)
11. [Agent Skills (SKILL.md)](#agent-skills-skillmd)
12. [CI Workflow (`ci.yml`)](#ci-workflow-ciyml)
13. [Implementation Order](#implementation-order)

---

## Goals & Consumption Modes

CI must ensure the extension is **buildable, testable, and installable from the git repo** for all five consumption modes. Registry publishing (PyPI, NPM) is out of scope — see `cd_and_distribution.md`.

| # | Mode | Install Command | What It Needs |
|---|------|----------------|---------------|
| 1 | **Python** | `pip install git+https://github.com/user/sqlite-muninn.git` | `pyproject.toml` with build backend, compiled binary |
| 2 | **Node.js / TypeScript** | `npm install git+https://github.com/user/sqlite-muninn.git` | `package.json`, compiled binary, ESM + CJS + types |
| 3 | **Browser (WASM)** | `<script>` or bundler import | Emscripten build: SQLite + muninn statically linked |
| 4 | **SQLite CLI** | `.load ./muninn` | Compiled `.so`/`.dylib`/`.dll` |
| 5 | **C library** | Download amalgamation, compile | Single `muninn.c` + `muninn.h` |

**Key insight:** Modes 1 and 2 work from git without any registry. `pip install` and `npm install` both support git URLs — CI just needs to ensure the packaging structure is correct.

---

## Version Support Policy

**Latest LTS only.** When a new LTS version is released, the previous LTS is immediately dropped from official support — even if it's still within its upstream support window.

| Runtime | Supported Version | Next Drop |
|---------|------------------|-----------|
| Python | 3.13 | When 3.14 releases (~Oct 2026) |
| Node.js | 22 LTS | When 24 LTS releases (~Oct 2026) |
| SQLite | System/Homebrew (3.37+) | N/A — test current + latest only |

**Rationale:** This keeps the CI matrix small and avoids compatibility shims. Users on older versions can still build from source — we just don't test or debug issues on those versions.

**What "not officially supported" means:**
- CI does not test against it
- Bug reports on older versions will be closed with "please upgrade"
- The code might still work — we just don't guarantee it

---

## Cross-Platform Build Matrix

Because muninn is a zero-dependency C11 library, every target can be built natively on its own CI runner — no cross-compilation needed.

| Target | Runner | Output | Notes |
|--------|--------|--------|-------|
| Linux x86_64 | `ubuntu-22.04` | `muninn.so` | Primary target |
| Linux ARM64 | `ubuntu-22.04-arm` | `muninn.so` | Native ARM64 runner |
| macOS Universal | `macos-15` | `muninn.dylib` | Fat binary: arm64 + x86_64 via `lipo` |
| Windows x86_64 | `windows-2022` | `muninn.dll` | MSVC build |

### macOS: Universal Binary from Single Runner

Apple's Clang can cross-compile between architectures. Build both on one ARM64 runner and combine:

```bash
# Build both architectures
cc -arch arm64  -O2 -std=c11 -fPIC -dynamiclib -undefined dynamic_lookup \
   -Isrc -o muninn_arm64.dylib src/*.c -lm

cc -arch x86_64 -O2 -std=c11 -fPIC -dynamiclib -undefined dynamic_lookup \
   -Isrc -o muninn_x86_64.dylib src/*.c -lm

# Combine into universal binary
lipo -create muninn_arm64.dylib muninn_x86_64.dylib -output muninn.dylib
```

### Windows: MSVC Build

The Makefile uses Unix conventions. Windows uses a separate MSVC build command:

```bat
cl.exe /O2 /MT /W4 /LD /Isrc ^
  src\muninn.c src\hnsw_vtab.c src\hnsw_algo.c ^
  src\graph_tvf.c src\node2vec.c src\vec_math.c ^
  src\priority_queue.c src\id_validate.c ^
  /Fe:muninn.dll
```

| Flag | Purpose |
|------|---------|
| `/MT` | Static CRT — zero runtime dependencies |
| `/LD` | Create DLL |

### Makefile Changes Needed

```makefile
# Cross-compilation support (macOS)
ifeq ($(UNAME_S),Darwin)
    ifdef ARCH
        CFLAGS_BASE += -arch $(ARCH)
    endif
    CFLAGS_BASE += -mmacosx-version-min=11.0
endif
```

---

## SQLite Header Vendoring

The current Makefile hard-codes the Homebrew path on macOS. For CI and cross-platform builds, vendor the SQLite amalgamation headers.

### `scripts/vendor_sqlite.sh`

```bash
#!/bin/bash
set -euo pipefail
SQLITE_VERSION="3510000"
SQLITE_YEAR="2025"
wget "https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip"
unzip -o sqlite-amalgamation-*.zip
cp sqlite-amalgamation-*/sqlite3.h sqlite-amalgamation-*/sqlite3ext.h vendor/
rm -rf sqlite-amalgamation-*
```

### Improved SQLite Detection (Makefile)

```makefile
# Try vendored headers first, then pkg-config, then Homebrew, then system
ifneq ($(wildcard vendor/sqlite3.h),)
    SQLITE_CFLAGS ?= -Ivendor
    SQLITE_LIBS ?= -lsqlite3
else
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
endif
```

This gives consumers three ways to point at SQLite:
1. **Vendored headers** — checked in or generated by `scripts/vendor_sqlite.sh`
2. **pkg-config** — auto-detected on Linux
3. **Homebrew** — auto-detected on macOS (current behavior preserved)
4. **Manual override** — `make all SQLITE_CFLAGS="-I/path" SQLITE_LIBS="-L/path -lsqlite3"`

---

## Packaging: Python

### Goal

`pip install git+https://github.com/user/sqlite-muninn.git` should just work.

### Package Name

`sqlite-muninn` → `import sqlite_muninn`

### Package Structure

```
sqlite_muninn/
    __init__.py          # loadable_path() + load() + version
    muninn.so            # Linux (or .dylib on macOS, .dll on Windows)
```

### API Surface

```python
# sqlite_muninn/__init__.py
import os
import sqlite3

__version__ = "0.1.0"

def loadable_path() -> str:
    """Return path to the muninn loadable extension (without file extension).
    SQLite's load_extension() automatically appends .so/.dylib/.dll.
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "muninn"))

def load(conn: sqlite3.Connection) -> None:
    """Load muninn into the given SQLite connection."""
    conn.load_extension(loadable_path())
```

### User-Facing Usage

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)
```

### Wheel Tags

Since muninn loads via `sqlite3.load_extension()` (not Python's C extension mechanism), it does **not** link against `libpython`:

```
sqlite_muninn-0.1.0-py3-none-{platform}.whl
```

- `py3` — any Python 3
- `none` — no Python ABI dependency
- `{platform}` — platform-specific (contains native binary)

### pyproject.toml (for git installs)

The repo needs a `pyproject.toml` that can build wheels from source. A custom build backend or a build script that compiles the C extension and places it alongside `__init__.py`.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sqlite-muninn"
version = "0.1.0"
description = "HNSW vector search + graph traversal + Node2Vec for SQLite"
requires-python = ">=3.13"
license = "MIT"

[tool.setuptools.packages.find]
include = ["sqlite_muninn*"]

[tool.setuptools.package-data]
sqlite_muninn = ["*.so", "*.dylib", "*.dll", "skills/**/*.md"]
```

**CI validates:** `pip install .` in a fresh venv succeeds, and `import sqlite_muninn; sqlite_muninn.load(conn)` works.

### macOS Caveat

Apple's system SQLite is compiled with `SQLITE_OMIT_LOAD_EXTENSION`. Users must:
- Install Python via Homebrew (`brew install python`)
- Or install `pysqlite3-binary`

This must be documented in README and package description.

---

## Packaging: Node.js / TypeScript

### Goal

`npm install git+https://github.com/user/sqlite-muninn.git` should just work (builds from source via `install` script).

For registry distribution (covered in CD doc), use the platform-specific `optionalDependencies` pattern (esbuild/sqlite-vec pattern).

### Package Structure (git install)

```
package.json          # At repo root, with "install" script that builds
npm/
  sqlite-muninn/
    index.mjs         # ESM entry
    index.cjs         # CJS entry
    index.d.ts        # TypeScript declarations
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

const EXT_MAP = { darwin: "dylib", linux: "so", win32: "dll" };

export function getLoadablePath() {
  const ext = EXT_MAP[platform];
  if (!ext) {
    throw new Error(`Unsupported platform: ${platform}. Supported: ${Object.keys(EXT_MAP).join(", ")}`);
  }
  // When installed from git, the binary is at the repo root
  const loadablePath = join(__dirname, "..", `muninn.${ext}`);
  if (!statSync(loadablePath, { throwIfNoEntry: false })) {
    throw new Error(`muninn binary not found at ${loadablePath}. Run 'make all' first.`);
  }
  return loadablePath;
}

export function load(db) {
  db.loadExtension(getLoadablePath());
}
```

### Compatible SQLite Drivers

The `Db` interface is deliberately minimal:

| Driver | Usage |
|--------|-------|
| `better-sqlite3` | `const db = new Database(":memory:"); load(db);` |
| `node:sqlite` (Node 22.5+) | `const db = new DatabaseSync(":memory:", { allowExtension: true }); load(db);` |
| `bun:sqlite` | `const db = new Database(":memory:"); load(db);` |

### CI Validates

- `npm install` from git URL builds the extension
- `node -e "const m = require('sqlite-muninn'); ..."` loads successfully
- TypeScript types are correct (`tsc --noEmit` on a test consumer)

---

## Packaging: WASM (Browser)

### Goal

Compile SQLite + muninn to WebAssembly for browser and edge runtimes. Extensions cannot be dynamically loaded in WASM — they must be statically linked.

### Static Registration Entry Point

```c
// src/sqlite3_wasm_extra_init.c
#include "sqlite3.h"
extern int sqlite3_muninn_init(sqlite3*, char**, const sqlite3_api_routines*);

int sqlite3_wasm_extra_init(const char *z) {
    return sqlite3_auto_extension((void(*)(void))sqlite3_muninn_init);
}
```

### Build Process (Emscripten)

```bash
# scripts/build_wasm.sh
set -euo pipefail

# Requires: amalgamation already generated at dist/muninn.c
# Requires: SQLite amalgamation at vendor/sqlite3.c

emcc -O2 -s WASM=1 -s EXPORTED_FUNCTIONS='["_sqlite3_open", ...]' \
     -DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_JSON1 \
     vendor/sqlite3.c \
     dist/muninn.c \
     src/sqlite3_wasm_extra_init.c \
     -o dist/muninn_sqlite3.js
```

### Output

```
dist/
    muninn_sqlite3.js       # JS glue code
    muninn_sqlite3.wasm     # WebAssembly binary
```

Platform-independent — a single build works everywhere.

### Performance Trade-off

WASM builds lose the native SIMD-accelerated distance functions in `vec_math.c`. Emscripten supports WASM SIMD (`-msimd128`) which can recover some performance — investigate during implementation.

### CI Validates

- Emscripten build completes without errors
- WASM binary loads in Node.js (basic smoke test)
- HNSW + graph TVFs work in WASM environment

---

## Packaging: C Library (Amalgamation)

### Goal

Ship a single `muninn.c` + `muninn.h` that any C/C++ project can compile and link. This is the primary distribution format for the C ecosystem — it's how SQLite itself distributes.

### Why Amalgamation Is Preferred

- Zero build-system coupling — works with Make, CMake, Meson, Zig, anything
- Single-translation-unit compilation enables 5-10% better optimization
- Trivial vendoring — just copy two files

### Amalgamation Script

```bash
# scripts/amalgamate.sh
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

### Makefile Target

```makefile
amalgamation: dist/muninn.c dist/muninn.h   ## Create amalgamation

dist/muninn.c dist/muninn.h: $(SRC) $(wildcard src/*.h)
	bash scripts/amalgamate.sh
```

### Consumer Usage

```bash
# Build as loadable extension
gcc -O2 -fPIC -shared muninn.c -o muninn.so -lm           # Linux
cc -O2 -fPIC -dynamiclib muninn.c -o muninn.dylib -lm     # macOS

# Or compile into an application with static linking
gcc -O2 myapp.c muninn.c -lsqlite3 -lm -o myapp
```

### Additional C Distribution Channels

| Channel | Effort | Reach |
|---------|--------|-------|
| **`make install` + pkg-config** | Low — Makefile targets + `.pc` template | Unix developers |
| **CMakeLists.txt** | Medium — `FetchContent` + `find_package` | CMake users |
| **Git submodule** | Zero — just pin a commit | Any build system |

These are detailed in the appendix but secondary to the amalgamation.

### CI Validates

- `make amalgamation` produces `dist/muninn.c` and `dist/muninn.h`
- Amalgamation compiles as a loadable extension on Linux
- Amalgamation compiles with static linking into a test binary

---

## Packaging: SQLite CLI Extension

### Goal

`sqlite3 :memory: ".load ./muninn"` just works after building.

### How It Works Today

This already works — `make all` produces `muninn.so`/`.dylib`/`.dll` which is directly loadable by the SQLite CLI.

### CI Validates

- Build extension on each platform
- Run: `sqlite3 :memory: ".load ./muninn" "SELECT 1"`
- Verify all three subsystems register (HNSW, graph TVFs, Node2Vec)

---

## Testing Strategy

### Test Pyramid

| Layer | What | Where | Runs On |
|-------|------|-------|---------|
| **C unit tests** | vec_math, priority_queue, hnsw_algo, id_validate | `make test` | All platforms |
| **Python integration** | Full extension via `sqlite3.load_extension()` | `make test-python` | Linux + macOS |
| **Sanitizers** | Memory errors, undefined behavior | `make debug` + tests | Linux only |
| **WASM smoke test** | Basic load + query in Node.js WASM | Custom script | Linux only |
| **Package install test** | `pip install .` and `npm install .` from local | CI scripts | Linux + macOS |

### Sanitizer Jobs (Linux Only)

```yaml
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
    - name: Build and test with sanitizers
      run: |
        CC=clang CFLAGS_EXTRA="${{ matrix.sanitizer.flags }} -g -O1 -fno-omit-frame-pointer" make test
```

### SQLite Version Testing

Test against current system SQLite and latest stable only. We don't support old SQLite versions — the "latest LTS only" philosophy extends here.

```yaml
sqlite-compat:
  runs-on: ubuntu-22.04
  strategy:
    matrix:
      sqlite:
        - { version: "system", label: "System (runner default)" }
        - { version: "3.51.0", label: "Latest stable" }
  name: SQLite ${{ matrix.sqlite.label }}
```

---

## Agent Skills (SKILL.md)

### Why This Is Part of CI

The `skills/` directory ships machine-readable documentation that AI coding tools consume. Stale or incorrect examples in `SKILL.md` cause AI tools to generate broken code — the #1 way users encounter muninn for the first time goes wrong.

**CI must validate that every code example in the skill files actually works.**

### Skill Directory Structure

```
skills/
    muninn/
        SKILL.md                      # YAML frontmatter + usage guide
        references/
            cookbook-python.md         # Python patterns
            cookbook-node.md           # Node.js patterns
            cookbook-c.md              # C/C++ patterns
            cookbook-sql.md            # Pure SQL patterns
            vector-encoding.md        # Cross-language vector format
            platform-caveats.md       # macOS, Windows, glibc
.claude-plugin/
    marketplace.json                  # Claude Code plugin manifest
```

### SKILL.md Validation Script

```python
# scripts/validate_skill_examples.py
# 1. Recursively find all .md files under skills/muninn/
# 2. Parse fenced code blocks (```sql, ```python, ```javascript)
# 3. Run SQL blocks against a fresh SQLite connection with muninn loaded
# 4. Run Python blocks in a subprocess with muninn importable
# 5. Report any blocks that error out (with file path and line number)
```

### CI Job

```yaml
validate-skills:
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
    - run: make all
    - uses: actions/setup-python@v5
      with:
        python-version: "3.13"
    - run: pip install pytest
    - run: python scripts/validate_skill_examples.py skills/muninn/
```

### Make Target

```makefile
skill: skills/muninn/SKILL.md                ## Stamp version into skill files
	@VERSION=$$(cat VERSION); \
	for f in skills/muninn/SKILL.md skills/muninn/references/*.md; do \
	    sed "s/{{VERSION}}/$$VERSION/g" "$$f" > "dist/$${f}"; \
	done
```

### Distribution Integration

The `skills/` directory must ship inside every package:

| Package | Included Skills Files |
|---------|----------------------|
| Python (PyPI) | `SKILL.md` + `cookbook-python.md` + `vector-encoding.md` + `platform-caveats.md` |
| Node.js (NPM) | `SKILL.md` + `cookbook-node.md` + `vector-encoding.md` + `platform-caveats.md` |
| C (Amalgamation) | `SKILL.md` + `cookbook-c.md` + `cookbook-sql.md` + `vector-encoding.md` |

---

## CI Workflow (`ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ── Build & Test (All Platforms) ─────────────────────────
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-22.04
            name: Linux x86_64
          - os: ubuntu-22.04-arm
            name: Linux ARM64
          - os: macos-15
            name: macOS (Universal)
          - os: windows-2022
            name: Windows x86_64
    runs-on: ${{ matrix.os }}
    name: Build (${{ matrix.name }})
    steps:
      - uses: actions/checkout@v4

      - name: Build extension
        if: runner.os != 'Windows'
        run: make all

      - name: Build extension (Windows)
        if: runner.os == 'Windows'
        uses: ilammy/msvc-dev-cmd@v1
      - if: runner.os == 'Windows'
        shell: cmd
        run: cl.exe /O2 /MT /W4 /LD /Isrc src\*.c /Fe:muninn.dll

      - name: Run C unit tests
        if: runner.os != 'Windows'
        run: make test

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Python integration tests
        if: runner.os != 'Windows'
        run: |
          pip install pytest
          python -m pytest pytests/ -v

      - name: SQLite CLI smoke test
        if: runner.os != 'Windows'
        run: |
          sqlite3 :memory: ".load ./muninn" "SELECT 1"

      - uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.name }}
          path: muninn.*

  # ── Sanitizers (Linux only) ──────────────────────────────
  sanitize:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        sanitizer:
          - { name: "ASan+UBSan", flags: "-fsanitize=address,undefined" }
    name: Sanitize (${{ matrix.sanitizer.name }})
    steps:
      - uses: actions/checkout@v4
      - run: |
          CC=clang CFLAGS_EXTRA="${{ matrix.sanitizer.flags }} -g -O1 -fno-omit-frame-pointer" make test

  # ── Package Install Tests ────────────────────────────────
  package-install:
    needs: build
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Test pip install from source
        run: |
          make all
          pip install .
          python -c "import sqlite_muninn; print(sqlite_muninn.__version__)"

      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - name: Test npm install from source
        run: |
          npm install .
          node -e "const m = require('sqlite-muninn'); console.log('loaded')"

  # ── WASM Build ───────────────────────────────────────────
  wasm:
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

  # ── Agent Skills Validation ──────────────────────────────
  validate-skills:
    needs: build
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: make all
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install pytest
      - run: python scripts/validate_skill_examples.py skills/muninn/
```

---

## Implementation Order

All items below are CI-scope only. Registry publishing is in `cd_and_distribution.md`.

### Phase 1: Build Infrastructure

1. **Add `VERSION` file** — single source of truth: `0.1.0`
2. **Vendor SQLite headers** — `scripts/vendor_sqlite.sh` + `vendor/` directory
3. **Improve SQLite detection** — vendored → pkg-config → Homebrew → manual chain
4. **Add Windows MSVC support** — `scripts/build_windows.bat`
5. **Add macOS universal binary support** — `ARCH` variable in Makefile + `lipo`

### Phase 2: CI Workflow

6. **Create `.github/workflows/ci.yml`** — build + test on 4 runners
7. **Add sanitizer job** — ASan+UBSan on Linux
8. **Add SQLite CLI smoke test** — `.load` + basic query on each platform

### Phase 3: Package Structure

9. **Create `sqlite_muninn/` Python package** — `__init__.py` + build integration
10. **Create `pyproject.toml`** — for `pip install git+...` from source
11. **Create `npm/sqlite-muninn/`** — `index.mjs`, `index.cjs`, `index.d.ts`
12. **Create `package.json`** at repo root — for `npm install git+...` from source
13. **Add package install CI jobs** — validate pip + npm install from source

### Phase 4: Amalgamation

14. **Create `scripts/amalgamate.sh`** — produces `dist/muninn.c` + `dist/muninn.h`
15. **Add `make amalgamation` target**
16. **Add amalgamation compile test** — CI builds from amalgamation

### Phase 5: WASM

17. **Create `src/sqlite3_wasm_extra_init.c`** — static extension registration
18. **Create `scripts/build_wasm.sh`** — Emscripten build
19. **Add WASM CI job** — build + smoke test
20. **Investigate WASM SIMD** (`-msimd128`)

### Phase 6: Agent Skills

21. **Create `skills/muninn/SKILL.md`** — YAML frontmatter + usage guide
22. **Create `skills/muninn/references/`** — per-language cookbooks
23. **Create `.claude-plugin/marketplace.json`**
24. **Write `scripts/validate_skill_examples.py`** — executable documentation validation
25. **Add skills validation CI job**
26. **Add `make skill` target** — version stamping

---

## Appendix: Additional C Distribution Channels

### `make install` with pkg-config

```
muninn.pc.in:
    prefix=@PREFIX@
    libdir=${prefix}/lib
    includedir=${prefix}/include
    Name: muninn
    Version: @VERSION@
    Requires: sqlite3
    Libs: -L${libdir} -lmuninn -lm
    Cflags: -I${includedir}
```

```makefile
PREFIX ?= /usr/local
install: muninn$(EXT) muninn.pc
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 muninn$(EXT) $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/include
	install -m 644 src/muninn.h $(DESTDIR)$(PREFIX)/include/
```

### CMakeLists.txt (for FetchContent consumers)

```cmake
cmake_minimum_required(VERSION 3.14)
project(muninn VERSION 0.1.0 LANGUAGES C)
find_package(SQLite3 REQUIRED)

add_library(muninn
    src/muninn.c src/hnsw_vtab.c src/hnsw_algo.c
    src/graph_tvf.c src/node2vec.c src/vec_math.c
    src/priority_queue.c src/id_validate.c
)
target_include_directories(muninn PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(muninn PUBLIC SQLite::SQLite3 PRIVATE m)
target_compile_features(muninn PUBLIC c_std_11)
```

Consumer:
```cmake
include(FetchContent)
FetchContent_Declare(muninn
    GIT_REPOSITORY https://github.com/user/sqlite-muninn.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(muninn)
target_link_libraries(my_app PRIVATE muninn)
```
