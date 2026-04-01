# src/ — C Source Conventions

## File Naming

### Prefix Convention

Files are grouped by prefix indicating their subsystem:

| Prefix | Subsystem | Depends on llama.cpp | WASM-excluded |
|--------|-----------|---------------------|---------------|
| `hnsw_` | HNSW vector index | No | No |
| `graph_` | Graph traversal, centrality, community, adjacency | No | No |
| `llama_` | LLM inference (chat, embed, label groups) | **Yes** | **Yes** |
| `llama_er` | Entity resolution (uses HNSW + graph + llama) | **Yes** | **Yes** |
| (none) | Shared utilities (`id_validate`, `vec_math`, `string_sim`, `priority_queue`, `node2vec`) | No | No |

**Rule: Any file that depends on llama.cpp MUST use the `llama_` prefix.** This is enforced by the build system — `scripts/generate_build.py` auto-excludes `llama_*.c` from WASM lite builds via glob. A file without the prefix that links llama.cpp symbols will break WASM compilation.

Note: `llama_er.c` depends on HNSW and graph subsystems in addition to llama.cpp. The `llama_` prefix correctly signals the llama.cpp dependency for WASM exclusion.

### SQLite Extension Header

Every `.c` file in `src/` that calls SQLite API functions MUST use:

```c
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3
```

**NOT** `#include <sqlite3.h>`. The extension API routes through a function pointer table (`sqlite3_api_routines`). Using `<sqlite3.h>` directly causes segfaults because the function pointers are NULL outside `.load` context.

The only exception is `muninn.c` (the entry point), which uses `SQLITE_EXTENSION_INIT1`.

### Header Guards

Use the pattern `FILENAME_H` matching the filename in uppercase:

```c
#ifndef LLAMA_LABEL_GROUPS_H
#define LLAMA_LABEL_GROUPS_H
// ...
#endif /* LLAMA_LABEL_GROUPS_H */
```

## Registration Pattern

Each subsystem exposes a single registration function called from `muninn.c`:

```c
// In header:
int subsystem_register_functions(sqlite3 *db);  // for scalar/aggregate functions
int subsystem_register_module(sqlite3 *db);      // for virtual tables
int subsystem_register_tvfs(sqlite3 *db);        // for table-valued functions
```

Registration functions that depend on llama.cpp MUST be inside the `#ifndef MUNINN_NO_LLAMA` guard in `muninn.c`.

## SQL Function Naming

All SQL-visible names use the `muninn_` prefix:

- Scalar functions: `muninn_chat()`, `muninn_embed()`, `muninn_extract_er()`
- Virtual tables: `muninn_label_groups`, `muninn_models`, `muninn_chat_models`
- Graph TVFs use `graph_` prefix: `graph_bfs`, `graph_leiden`, `graph_closeness`
- HNSW uses `hnsw_index` (legacy, predates muninn_ convention)

The SQL-facing name does NOT need to match the C filename. For example, `llama_label_groups.c` registers the SQL virtual table as `muninn_label_groups`.

## Dynamic Table/Column Names

Any SQL that interpolates user-provided table or column names MUST validate them with `id_validate()` first (from `id_validate.c`). This prevents SQL injection. Validated names are safe to use in `[bracket]` quoting:

```c
if (id_validate(table_name) != 0) return SQLITE_ERROR;
snprintf(sql, sizeof(sql), "SELECT [%s] FROM [%s]", col_name, table_name);
```

## Memory Management

- Use `sqlite3_malloc()`/`sqlite3_free()` for vtab structs (SQLite manages their lifecycle)
- Use `malloc()`/`free()`/`calloc()`/`realloc()` for internal data structures
- Use `strdup()` for string copies in internal structs
- Use `sqlite3_mprintf()` for error messages assigned to `vtab->base.zErrMsg`

## Virtual Table xBestIndex

`argvIndex` values MUST be contiguous (1, 2, 3...) with no gaps. SQLite returns "virtual table malfunction" if there are gaps. Use a two-pass approach:
1. Pass 1: scan constraints, record which columns have usable EQ constraints
2. Pass 2: assign sequential argvIndex values in column order
3. Communicate constraint presence via `idxNum` bitmask

## Error Reporting

Virtual table methods set errors via:
```c
vtab->base.zErrMsg = sqlite3_mprintf("muninn_xxx: descriptive message");
return SQLITE_ERROR;
```

Scalar functions set errors via:
```c
sqlite3_result_error(ctx, "muninn_xxx: descriptive message", -1);
```

## Testing

- C unit tests live in `test/test_<module>.c`
- Test framework: `test_common.h` with `ASSERT`, `ASSERT_EQ_INT`, `RUN_TEST`, `TEST` macros
- New test suites: add `test/test_<module>.c`, extern in `test_main.c`, add to `TEST_SRC` in Makefile
- `sqlite3_api` is NULL outside `.load` — tests use `sqlite3_auto_extension()` to capture the real API table
- Test helpers gated behind `#ifdef MUNINN_TESTING` (set by test build)
