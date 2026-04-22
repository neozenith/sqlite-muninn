---
name: muninn-troubleshoot
description: >
  Diagnoses muninn build failures, extension-loading errors, platform-specific
  pitfalls, and runtime issues across SQLite CLI, C, Python, Node.js, and WASM.
  Covers SQLITE_OMIT_LOAD_EXTENSION on macOS system Python, CMake hangs on
  Apple Silicon (GGML_NATIVE), Metal GPU toggles (MUNINN_GPU_LAYERS), llama.cpp
  log verbosity (MUNINN_LOG_LEVEL), ASan-built dylibs failing in Python, GGUF
  model resolution, and wasm32 vs wasm64 linkage. Use when the user mentions
  "unable to open shared library", "not authorized", "SQLITE_OMIT_LOAD_EXTENSION",
  "enable_load_extension fails", "muninn build failed", "CMake hangs",
  "Segmentation fault muninn", "model not registered", "MUNINN_GPU_LAYERS",
  "WASM muninn fails", "muninn troubleshooting", or hits a muninn load / build
  error.
license: MIT
---

# muninn-troubleshoot — Diagnose build and runtime failures

Every issue below is something that has actually broken in development or user-reported. Work through the symptom → diagnosis → fix pattern rather than rewriting the user's code blindly.

## Extension fails to load

### Python: `enable_load_extension` raises or returns `not authorized`

**Symptom**
```
AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'
# or
sqlite3.OperationalError: not authorized
```

**Diagnosis** — macOS system Python (`/usr/bin/python3`) and some Linux distro Pythons are compiled with `SQLITE_OMIT_LOAD_EXTENSION`. Check:

```python
import sqlite3
print(sqlite3.sqlite_version)
db = sqlite3.connect(":memory:")
print(hasattr(db, "enable_load_extension"))   # False → system Python blocked
```

**Fix** — one of:

```bash
# Option A: Homebrew Python
brew install python
/opt/homebrew/bin/python3 -m pip install sqlite-muninn

# Option B: pysqlite3-binary (drop-in replacement with a bundled non-OMIT SQLite)
pip install pysqlite3-binary
```

```python
# If using pysqlite3-binary
import pysqlite3 as sqlite3
```

### SQLite CLI: `Error: unable to open shared library`

**Symptom**
```
sqlite> .load ./muninn
Error: unable to open shared library ./muninn
```

**Diagnosis** — one of:
1. File doesn't exist at that path — check with `ls ./muninn*`
2. Architecture mismatch (arm64 binary on x86_64 or vice versa) — check `file ./muninn.dylib`
3. SIP / notarization on macOS — `xattr -l ./muninn.dylib` shows quarantine flag

**Fix**
```bash
# Remove quarantine on macOS
xattr -d com.apple.quarantine ./muninn.dylib

# Rebuild for current architecture
make clean && make all
```

Do not drop the file extension *in the Bash-visible path*: `.load ./muninn` is correct, `.load ./muninn.dylib` also works. Both resolve to the same file.

### Node.js: `Cannot open shared library` or undefined `load` export

**Diagnosis** — either `better-sqlite3` is missing, or you installed `sqlite-muninn` without native binaries (CI slim image, Alpine musl without glibc).

**Fix**
```bash
# Ensure both packages; rebuild bindings if on odd glibc
npm install sqlite-muninn better-sqlite3
npm rebuild better-sqlite3
```

On Alpine (musl) the prebuilt binaries may not match — build from source:
```bash
apk add build-base python3 cmake
npm install --build-from-source sqlite-muninn
```

## Build fails

### CMake hangs on Apple Silicon during first `make all`

**Symptom** — `make all` sits at `-- Detecting CXX compiler features` for 10+ minutes.

**Diagnosis** — llama.cpp defaults to `GGML_NATIVE=ON`, which runs `check_cxx_source_runs()` for SVE/SME CPU features. On Apple Silicon, this probe hangs indefinitely.

**Fix** — the project's `Makefile` already passes `-DGGML_NATIVE=OFF`. If you are bypassing the Makefile (calling CMake directly), pass the flag manually:

```bash
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_NATIVE=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON
```

### Linux: linker errors on `libstdc++` or `libpthread`

**Symptom**
```
undefined reference to `__pthread_create`
undefined reference to `std::basic_string`
```

**Diagnosis** — llama.cpp is C++, so the muninn extension needs to pull C++ runtime and pthreads when statically linked.

**Fix** — ensure these flags are in the link step (the Makefile handles this automatically):
```
-lstdc++ -lpthread
```

macOS equivalent: `-lc++ -framework Accelerate -framework Metal -framework MetalKit -framework Foundation`.

### WASM: invalid wasm64 objects

**Symptom** — llama.cpp produces `.o` files that the final link rejects as "not a wasm32 object."

**Diagnosis** — recent llama.cpp defaults to `LLAMA_WASM_MEM64=ON`, producing wasm64 objects. muninn targets wasm32 for compatibility with `@sqlite.org/sqlite-wasm`.

**Fix** — pass `-DLLAMA_WASM_MEM64=OFF` when configuring. The Makefile handles this for `make wasm`.

## Runtime errors

### `Error: no such function: muninn_embed`

**Diagnosis** — extension loaded but no embedding model registered. The function exists; the failure is from the model lookup inside it.

**Fix**
```sql
INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');
```

If the error is literally "no such function" (not "model not registered"), the extension failed to load — debug that first.

### `Error: model not registered: 'X'`

**Diagnosis** — model name mismatch, or the model was registered on a different connection.

**Fix** — list registered models:
```sql
SELECT name, dim FROM temp.muninn_models;
SELECT name FROM temp.muninn_chat_models;
```

Models are **session-scoped**. Reconnecting drops them. Re-register after `sqlite3.connect(...)`.

### Segmentation fault in Python tests

**Diagnosis** — you're loading `muninn` built with AddressSanitizer into a non-ASan Python interpreter. ASan must be present on both sides or neither.

**Fix**
```bash
make clean
make all           # not `make debug`
```

Use `make debug` only for the C test runner, never for integration from Python.

### Metal GPU disabled on macOS

**Diagnosis** — `MUNINN_GPU_LAYERS=0` was set, or the binary was built without Metal.

**Fix**
```bash
unset MUNINN_GPU_LAYERS                  # reverts to compile-time default (99 on macOS)
export MUNINN_LOG_LEVEL=warn              # surface llama.cpp warnings if still slow

# Verify at build time:
make clean
grep -r "GGML_METAL=ON" scripts/generate_build.py
```

On Linux there is no CUDA support in the current build; GPU layers are always 0.

### llama.cpp is silent (can't see warnings)

**Fix**
```bash
export MUNINN_LOG_LEVEL=verbose   # all logs
export MUNINN_LOG_LEVEL=warn      # warnings + errors
export MUNINN_LOG_LEVEL=error     # errors only (default is silent)
```

## WASM-specific pitfalls

### Model file cannot be found

WASM reads from the virtual filesystem, not HTTP. Preload:

```javascript
const response = await fetch("/models/miniLM.gguf");
const bytes = new Uint8Array(await response.arrayBuffer());
sqlite3.FS.writeFile("/models/miniLM.gguf", bytes);

db.exec(`
  INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('/models/miniLM.gguf');
`);
```

For persistence across page reloads, use OPFS (`sqlite3.oo1.OpfsDb`) and preload models into OPFS once.

### Slow embedding / chat in the browser

Expected — WASM is CPU-only with no Metal. For interactive demos, use a small model (MiniLM 384d) and keep batch size 1. For production quality, run the native build behind an API instead.

## Diagnostic snippets

```sql
-- Confirm each subsystem registered
SELECT name FROM pragma_function_list WHERE name LIKE 'muninn_%';
SELECT name FROM pragma_module_list   WHERE name LIKE 'hnsw_%' OR name LIKE 'graph_%' OR name = 'muninn_models' OR name = 'muninn_chat_models';

-- Check loaded extension compile options
SELECT sqlite_version();
SELECT sqlite_compileoption_used('ENABLE_LOAD_EXTENSION');   -- must be 1
```

```bash
# Show which shared library SQLite loaded (macOS)
sqlite3 -cmd ".load ./muninn" ":memory:" "SELECT 1;" 2>&1
lsof -p $$ | grep muninn
```

## When to escalate

If none of the above applies:
1. Capture `MUNINN_LOG_LEVEL=verbose` output.
2. Run `make test` — if C tests fail, the issue is in the extension itself.
3. Run `make test-python` — if Python tests fail but C passes, the issue is the binding layer.
4. Open an issue at https://github.com/neozenith/sqlite-muninn/issues with platform + architecture + log tail.

## See also

- [muninn-setup](../muninn-setup/SKILL.md) — original install/load paths
- [getting-started.md § Common pitfalls](../../docs/getting-started.md) — authoritative list
- Project `Makefile` — canonical build flags
