#!/bin/bash
# Build muninn + SQLite as a WASM module via Emscripten.
#
# Prerequisites:
#   - Emscripten SDK (emsdk) activated
#   - Amalgamation built: make amalgamation (produces dist/muninn.c)
#   - SQLite amalgamation: vendor/sqlite3.c (download via scripts/vendor_sqlite.sh)
#
# Output:
#   dist/muninn_sqlite3.js    — JS glue code
#   dist/muninn_sqlite3.wasm  — WebAssembly binary
set -euo pipefail

OUTDIR=dist
mkdir -p "$OUTDIR"

# Check prerequisites
if [ ! -f "$OUTDIR/muninn.c" ]; then
    echo "Error: amalgamation not found at $OUTDIR/muninn.c"
    echo "Run: make amalgamation"
    exit 1
fi

# Download SQLite amalgamation if not present
SQLITE_SRC="$OUTDIR/sqlite3.c"
if [ ! -f "$SQLITE_SRC" ]; then
    SQLITE_VERSION="${SQLITE_VERSION:-3510000}"
    SQLITE_YEAR="${SQLITE_YEAR:-2025}"
    echo "Downloading SQLite amalgamation ${SQLITE_VERSION}..."
    curl -sL "https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip" -o /tmp/sqlite.zip
    unzip -o /tmp/sqlite.zip -d /tmp/sqlite_amal
    cp /tmp/sqlite_amal/sqlite-amalgamation-*/sqlite3.c "$SQLITE_SRC"
    cp /tmp/sqlite_amal/sqlite-amalgamation-*/sqlite3.h "$OUTDIR/sqlite3.h"
    cp /tmp/sqlite_amal/sqlite-amalgamation-*/sqlite3ext.h "$OUTDIR/sqlite3ext.h"
    rm -rf /tmp/sqlite.zip /tmp/sqlite_amal
fi

echo "Building WASM..."

emcc -O2 \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_sqlite3_open","_sqlite3_close","_sqlite3_exec","_sqlite3_errmsg","_sqlite3_prepare_v2","_sqlite3_step","_sqlite3_finalize","_sqlite3_column_text","_sqlite3_column_int","_sqlite3_column_double","_sqlite3_column_blob","_sqlite3_column_bytes","_sqlite3_column_count","_sqlite3_column_name","_sqlite3_bind_text","_sqlite3_bind_int","_sqlite3_bind_double","_sqlite3_bind_blob","_sqlite3_bind_null","_sqlite3_reset","_sqlite3_free","_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall","UTF8ToString","stringToUTF8","getValue","setValue"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16777216 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createMuninnSQLite" \
    -DSQLITE_ENABLE_FTS5 \
    -DSQLITE_ENABLE_JSON1 \
    -DSQLITE_ENABLE_RTREE \
    -DSQLITE_THREADSAFE=0 \
    -DSQLITE_OMIT_LOAD_EXTENSION \
    -I"$OUTDIR" \
    "$SQLITE_SRC" \
    "$OUTDIR/muninn.c" \
    src/sqlite3_wasm_extra_init.c \
    -lm \
    -o "$OUTDIR/muninn_sqlite3.js"

echo "WASM build complete:"
ls -la "$OUTDIR/muninn_sqlite3.js" "$OUTDIR/muninn_sqlite3.wasm"
