#!/bin/bash
# Generate the muninn amalgamation: a single muninn.c + muninn.h
# that can be compiled standalone without the full source tree.
set -euo pipefail

VERSION=$(cat VERSION 2>/dev/null || echo "0.0.0")
OUTDIR=dist
OUT="${OUTDIR}/muninn.c"

mkdir -p "$OUTDIR"

# ── Header ordering matters: dependencies before dependents ──
INTERNAL_HEADERS=(
    src/vec_math.h
    src/priority_queue.h
    src/hnsw_algo.h
    src/id_validate.h
    src/hnsw_vtab.h
    src/graph_common.h
    src/graph_tvf.h
    src/graph_load.h
    src/graph_centrality.h
    src/graph_community.h
    src/node2vec.h
    src/muninn.h
)

# ── Source ordering: dependencies before dependents ──
SOURCES=(
    src/vec_math.c
    src/priority_queue.c
    src/hnsw_algo.c
    src/id_validate.c
    src/hnsw_vtab.c
    src/graph_tvf.c
    src/graph_load.c
    src/graph_centrality.c
    src/graph_community.c
    src/node2vec.c
    src/muninn.c
)

# ── Write file header ──
cat > "$OUT" <<HEADER
/*
 * muninn amalgamation — v${VERSION}
 * Generated $(date -u +%Y-%m-%d)
 *
 * HNSW vector search + graph traversal + Node2Vec for SQLite.
 * https://github.com/user/sqlite-muninn
 *
 * Build as loadable extension:
 *   gcc -O2 -fPIC -shared muninn.c -o muninn.so -lm           # Linux
 *   cc -O2 -fPIC -dynamiclib muninn.c -o muninn.dylib -lm     # macOS
 *   cl /O2 /MT /LD muninn.c /Fe:muninn.dll                    # Windows (MSVC)
 *
 * Or compile into your application:
 *   gcc -O2 myapp.c muninn.c -lsqlite3 -lm -o myapp
 */

/* Enable POSIX functions (strdup) on strict C11 compilers */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/* SQLite extension API — required for all builds */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <ctype.h>

HEADER

# ── Inline all internal headers (strip #include "..." lines) ──
for f in "${INTERNAL_HEADERS[@]}"; do
    if [ -f "$f" ]; then
        echo "/* ──── $f ──── */" >> "$OUT"
        # Remove #include "..." (internal headers) and header guards we'll handle
        grep -v '#include "' "$f" >> "$OUT"
        echo "" >> "$OUT"
    fi
done

# ── Inline all source files (strip #include "..." lines) ──
for f in "${SOURCES[@]}"; do
    if [ -f "$f" ]; then
        echo "/* ──── $f ──── */" >> "$OUT"
        # Remove internal #include "..." and SQLITE_EXTENSION_INIT1 (already at top)
        grep -v '#include "' "$f" | grep -v 'SQLITE_EXTENSION_INIT1' >> "$OUT"
        echo "" >> "$OUT"
    fi
done

# ── Copy public header ──
cp src/muninn.h "${OUTDIR}/muninn.h"

LINES=$(wc -l < "$OUT")
echo "Amalgamation: ${OUT} (${LINES} lines)"
echo "Header:       ${OUTDIR}/muninn.h"
