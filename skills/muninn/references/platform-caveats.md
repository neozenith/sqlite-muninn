# Platform Caveats

## macOS

**System Python's sqlite3 disables `load_extension()`.**

Apple's SQLite is compiled with `SQLITE_OMIT_LOAD_EXTENSION`. You'll get:
```
AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'
```

**Fix options:**
1. Install Python via Homebrew: `brew install python`
2. Install `pysqlite3-binary`: `pip install pysqlite3-binary`
3. Use the WASM build in a browser context

## Windows

The extension is built with MSVC (`/MT` flag) for zero runtime dependencies.
The resulting `muninn.dll` is fully self-contained.

Ensure the DLL architecture matches your Python/Node.js architecture (both must be x64).

## Linux (glibc)

Binaries built on `ubuntu-22.04` require glibc >= 2.35.
If you're on an older system (RHEL 8, Amazon Linux 2), build from source:

```bash
# From the amalgamation
gcc -O2 -fPIC -shared muninn.c -o muninn.so -lm
```

## SQLite Version Requirements

Minimum: SQLite 3.9.0 (2015) for table-valued functions.
Recommended: SQLite 3.38.0+ for optimal virtual table performance.

Key compile-time flags that break muninn:
- `SQLITE_OMIT_LOAD_EXTENSION` — disables all extension loading
- `SQLITE_OMIT_VIRTUALTABLE` — disables virtual tables entirely
