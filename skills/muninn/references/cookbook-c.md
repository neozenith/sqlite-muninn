# C/C++ Cookbook

## Loadable Extension

```c
#include <sqlite3.h>
#include <stdio.h>

int main(void) {
    sqlite3 *db;
    char *err = NULL;

    sqlite3_open(":memory:", &db);
    sqlite3_enable_load_extension(db, 1);

    int rc = sqlite3_load_extension(db, "./muninn", NULL, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to load: %s\n", err);
        sqlite3_free(err);
        return 1;
    }
    sqlite3_enable_load_extension(db, 0);

    // Now HNSW, graph TVFs, and node2vec are available
    sqlite3_exec(db, "SELECT 1", NULL, NULL, NULL);

    sqlite3_close(db);
    return 0;
}
```

## Static Linking (Amalgamation)

```c
#include "sqlite3.h"

// Forward-declare muninn's entry point
extern int sqlite3_muninn_init(sqlite3*, char**, const sqlite3_api_routines*);

int main(void) {
    // Register as auto-extension (before any sqlite3_open)
    sqlite3_auto_extension((void(*)(void))sqlite3_muninn_init);

    sqlite3 *db;
    sqlite3_open(":memory:", &db);
    // muninn is automatically loaded — no enable_load_extension needed

    sqlite3_close(db);
    return 0;
}
```

Build:
```bash
gcc -O2 myapp.c muninn.c sqlite3.c -lpthread -ldl -lm -o myapp
```

## Vector Binding

```c
// Prepare a vector
int dim = 128;
float *vec = malloc(dim * sizeof(float));
for (int i = 0; i < dim; i++) vec[i] = (float)i / dim;

// Bind as blob parameter
sqlite3_stmt *stmt;
sqlite3_prepare_v2(db,
    "INSERT INTO my_index(rowid, vector) VALUES (?, ?)", -1, &stmt, NULL);
sqlite3_bind_int64(stmt, 1, 42);
sqlite3_bind_blob(stmt, 2, vec, dim * sizeof(float), SQLITE_TRANSIENT);
sqlite3_step(stmt);
sqlite3_finalize(stmt);
free(vec);
```

## Reading Search Results

```c
sqlite3_stmt *stmt;
float query[128] = { /* ... */ };

sqlite3_prepare_v2(db,
    "SELECT rowid, distance FROM my_index WHERE vector MATCH ? AND k = 10",
    -1, &stmt, NULL);
sqlite3_bind_blob(stmt, 1, query, sizeof(query), SQLITE_STATIC);

while (sqlite3_step(stmt) == SQLITE_ROW) {
    int64_t id = sqlite3_column_int64(stmt, 0);
    double dist = sqlite3_column_double(stmt, 1);
    printf("id=%lld distance=%.4f\n", id, dist);
}
sqlite3_finalize(stmt);
```

## Thread Safety

- One `sqlite3*` connection per thread
- The HNSW index is connection-scoped (in-memory)
- File-backed databases can be shared via separate connections with WAL mode
