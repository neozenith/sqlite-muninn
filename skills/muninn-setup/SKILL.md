---
name: muninn-setup
description: >
  Installs, loads, and smoke-tests the muninn SQLite extension across SQLite CLI,
  C, Python, Node.js, and WASM runtimes. Covers pip install sqlite-muninn,
  npm install sqlite-muninn, prebuilt .dylib/.so/.dll from GitHub Releases, and
  from-source builds with make. Use when the user mentions "install muninn",
  "load muninn", "load the extension", "enable_load_extension", ".load ./muninn",
  "sqlite3_muninn_init", "sqlite-muninn", "pip install sqlite-muninn",
  "npm install sqlite-muninn", "muninn.dylib", "muninn WASM", or asks to get
  started with the library.
license: MIT
---

# muninn-setup — Install, load, and verify muninn

`muninn` is a single native shared library (`muninn.dylib` / `muninn.so` / `muninn.dll`). Every runtime loads the same binary. Pick the lane that matches the user's host environment, then run the smoke test at the bottom — if it returns rows, the extension works.

## Install matrix

| Runtime | Install | Load call |
|---------|---------|-----------|
| SQLite CLI | prebuilt binary from [GitHub Releases](https://github.com/neozenith/sqlite-muninn/releases), or `make all` from source | `.load ./muninn` (no file extension) |
| Python | `pip install sqlite-muninn` | `import sqlite_muninn; sqlite_muninn.load(db)` |
| Node.js | `npm install sqlite-muninn better-sqlite3` | `import { load } from "sqlite-muninn"; load(db);` |
| C | static link (`make all` → `muninn.dylib`), or `sqlite3_auto_extension(sqlite3_muninn_init)` | see C lane below |
| WASM | `npm install sqlite-muninn-wasm` (bundles `muninn.wasm` + `sqlite3.wasm`) | see WASM lane below |

## SQLite CLI lane

```bash
# prebuilt binary (fastest)
curl -L https://github.com/neozenith/sqlite-muninn/releases/latest/download/muninn-macos-arm64.dylib -o muninn.dylib
sqlite3

sqlite> .load ./muninn
sqlite> SELECT sqlite_version();
```

```text
sqlite_version()
----------------
3.45.1
```

Drop the file extension in `.load` — SQLite appends `.so`/`.dylib`/`.dll` for the current platform automatically.

## Python lane

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)        # resolves the bundled binary inside the wheel
db.enable_load_extension(False)  # re-secure the connection
```

macOS system Python (`/usr/bin/python3`) is compiled with `SQLITE_OMIT_LOAD_EXTENSION`. If `enable_load_extension` raises `AttributeError` or `not authorized`, switch to Homebrew Python (`brew install python`) or install `pysqlite3-binary` as a drop-in replacement — see [muninn-troubleshoot](../muninn-troubleshoot/SKILL.md).

## Node.js lane

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);                     // reads the bundled .node binding from the npm package
console.log(db.pragma("compile_options"));
```

`better-sqlite3` is required — `node:sqlite` (Node 22+) does not expose `loadExtension`.

## C lane

Two patterns depending on whether you want the extension auto-loaded for every `sqlite3_open` or only on demand.

```c
// Pattern A: auto-load for every new connection
#include <sqlite3.h>
extern int sqlite3_muninn_init(sqlite3 *, char **, const sqlite3_api_routines *);

int main(void) {
    sqlite3_auto_extension((void (*)(void))sqlite3_muninn_init);

    sqlite3 *db;
    sqlite3_open(":memory:", &db);
    // muninn is now loaded — hnsw_index, graph_*, muninn_* all available
    return 0;
}
```

```c
// Pattern B: explicit load of a prebuilt shared library
sqlite3_open(":memory:", &db);
sqlite3_enable_load_extension(db, 1);
sqlite3_load_extension(db, "./muninn", NULL, &err);
```

Link with the amalgamation (`dist/muninn.c`) from Releases, or link against the built `muninn.dylib`. When statically linking, add:

- **macOS:** `-lc++ -framework Accelerate -framework Metal -framework MetalKit -framework Foundation`
- **Linux:** `-lstdc++ -lpthread`

## WASM lane

```javascript
import { sqlite3InitModule } from "@sqlite.org/sqlite-wasm";
import muninnInit from "sqlite-muninn-wasm";

const sqlite3 = await sqlite3InitModule();
const db = new sqlite3.oo1.DB(":memory:", "c");
muninnInit(db);   // registers hnsw_index, graph_*, and muninn_embed (CPU-only)
```

WASM ships the same SQL surface but runs CPU-only (no Metal/BLAS). GGUF models must be preloaded into OPFS or MEMFS — `muninn_embed_model` reads from the virtual filesystem, not the network.

## Smoke test (every lane)

Run this after loading. If both queries return rows, the extension is wired up correctly.

```sql
.load ./muninn                              -- or the host-language equivalent above

-- HNSW subsystem
CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=4, metric='l2');
INSERT INTO vec(rowid, vector) VALUES (1, X'0000803F000000000000000000000000');
SELECT rowid, distance FROM vec
  WHERE vector MATCH X'0000803F000000000000000000000000' AND k = 1;

-- Graph subsystem (constraint-form TVF — no positional args)
CREATE TABLE edges (src TEXT, dst TEXT);
INSERT INTO edges VALUES ('a', 'b'), ('b', 'c');
SELECT node, depth FROM graph_bfs
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'a' AND max_depth = 5;
```

```text
rowid  distance
-----  --------
1      0.0

node  depth
----  -----
a     0
b     1
c     2
```

## Optional — pull a GGUF model for `muninn_embed` / `muninn_chat`

Skip this unless the user wants text embedding or LLM extraction. Once downloaded, see [muninn-embed-text](../muninn-embed-text/SKILL.md) or [muninn-chat-extract](../muninn-chat-extract/SKILL.md).

```bash
mkdir -p models
curl -L -o models/all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

| Model | Dims | Size | Good for |
|-------|------|------|----------|
| all-MiniLM-L6-v2 Q8_0 | 384 | 36 MB | Default English embedder |
| nomic-embed-text-v1.5 Q4_K_M | 768 | 84 MB | Long context (8192 tokens), multilingual |
| Qwen3-Embedding-8B Q4_K_M | 4096 | 4.7 GB | State-of-the-art retrieval quality |
| Qwen3.5-4B-Instruct Q4_K_M | — | ~2.6 GB | Chat + NER/RE extraction |

## Common pitfalls

- **`unable to open shared library`** — macOS system Python blocks extension loading. Use Homebrew Python or `pysqlite3-binary`.
- **`Segmentation fault`** in Python tests — you are loading an ASan-instrumented `make debug` build into a non-ASan Python. Run `make all` for integration builds.
- **`no such function: muninn_embed`** — the extension loaded but you haven't registered a model. Run `INSERT INTO temp.muninn_models(name, model) SELECT 'MiniLM', muninn_embed_model('models/...gguf');` first.
- **`CMake hangs on Apple Silicon` during `make all`** — the bundled Makefile already passes `-DGGML_NATIVE=OFF`. If you bypass it, pass the flag manually.

See [muninn-troubleshoot](../muninn-troubleshoot/SKILL.md) for deeper diagnostic workflows.

## See also

- [muninn-vector-search](../muninn-vector-search/SKILL.md) — next step once loaded
- [muninn-embed-text](../muninn-embed-text/SKILL.md) — if the user downloads a GGUF embedding model
- [getting-started.md](../../docs/getting-started.md) — full authoritative walkthrough
