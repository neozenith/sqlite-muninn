#!/usr/bin/env python3
"""Verify that pysqlite3 supports enable_load_extension.

Exits 0 if extension loading is available, 1 if not.
Used as a CI diagnostic step to fail fast with a clear message.
"""

import sys

import pysqlite3 as sqlite3

conn = sqlite3.connect(":memory:")
try:
    conn.enable_load_extension(True)
    print(f"OK: pysqlite3 (sqlite {sqlite3.sqlite_version}) has enable_load_extension")
except AttributeError:
    print(
        f"FAIL: pysqlite3 (sqlite {sqlite3.sqlite_version}) was compiled with "
        "SQLITE_OMIT_LOAD_EXTENSION — extension loading is disabled",
        file=sys.stderr,
    )
    sys.exit(1)
finally:
    conn.close()
