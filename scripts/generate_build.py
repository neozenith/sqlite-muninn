#!/usr/bin/env python3
"""Centralised build configuration for muninn.

Single source of truth for source file lists, cmake flags, library paths,
and all generated build artifacts. Source and header files are auto-discovered
via glob and dependency-sorted by parsing #include directives.

Subcommands:
    query <VAR>   Print a build variable value (for Makefile $(shell) evaluation)
    windows       Generate build/generated/build_windows.bat
    amalgamate    Generate dist/muninn.c + dist/muninn.h
    npm           Generate npm platform sub-packages
    version       Stamp VERSION into target files

Usage:
    python3 scripts/generate_build.py query SRC
    python3 scripts/generate_build.py amalgamate
    python3 scripts/generate_build.py windows --dry-run
"""

import argparse
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"


# ======================================================================
# STATIC CONFIGURATION (not discoverable from the filesystem)
# ======================================================================

# Subset of sources linked into test_runner (semantic choice — can't be auto-discovered)
TEST_LINK_SOURCES = [
    "src/vec_math.c",
    "src/priority_queue.c",
    "src/hnsw_algo.c",
    "src/id_validate.c",
    "src/graph_load.c",
    "src/graph_csr.c",
    "src/graph_selector_parse.c",
    "src/graph_selector_eval.c",
    "src/embed_gguf.c",
]

# ── CMake flags ──────────────────────────────────────────────────────
CMAKE_FLAGS_BASE = {
    "BUILD_SHARED_LIBS": "OFF",
    "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
    "GGML_NATIVE": "OFF",
    "GGML_METAL": "OFF",
    "GGML_CUDA": "OFF",
    "GGML_VULKAN": "OFF",
    "GGML_HIP": "OFF",
    "GGML_SYCL": "OFF",
    "GGML_OPENMP": "OFF",
    "GGML_BACKEND_DL": "OFF",
    "LLAMA_BUILD_COMMON": "OFF",
    "LLAMA_BUILD_TESTS": "OFF",
    "LLAMA_BUILD_TOOLS": "OFF",
    "LLAMA_BUILD_EXAMPLES": "OFF",
    "LLAMA_BUILD_SERVER": "OFF",
    "CMAKE_BUILD_TYPE": "MinSizeRel",
}

# WASM overrides: remove flags not relevant for emscripten, add WASM-specific
_WASM_REMOVE = {"GGML_HIP", "GGML_SYCL", "LLAMA_BUILD_COMMON", "LLAMA_BUILD_TOOLS"}
CMAKE_FLAGS_WASM = {k: v for k, v in CMAKE_FLAGS_BASE.items() if k not in _WASM_REMOVE}
CMAKE_FLAGS_WASM["LLAMA_WASM_MEM64"] = "OFF"

# ── llama.cpp paths ──────────────────────────────────────────────────
LLAMA_LIBS = ["libllama.a", "libggml.a", "libggml-base.a", "libggml-cpu.a"]
LLAMA_LIBS_WINDOWS = [
    r"vendor\llama.cpp\build\src\MinSizeRel\llama.lib",
    r"vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml.lib",
    r"vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml-base.lib",
    r"vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml-cpu.lib",
]
LLAMA_INCLUDE_DIRS = ["vendor/llama.cpp/include", "vendor/llama.cpp/ggml/include"]

# ── NPM platforms ────────────────────────────────────────────────────
NPM_PLATFORMS = {
    "darwin-arm64": ("darwin", "arm64", "dylib"),
    "darwin-x64": ("darwin", "x64", "dylib"),
    "linux-x64": ("linux", "x64", "so"),
    "linux-arm64": ("linux", "arm64", "so"),
    "win32-x64": ("win32", "x64", "dll"),
}

# ── Version stamp targets ───────────────────────────────────────────
# Each: (file relative to project root, regex pattern)
VERSION_STAMP_TARGETS = [
    (
        "skills/muninn/SKILL.md",
        r'(  version:\s*")[\d]+\.[\d]+\.[\d]+[^"]*(")',
    ),
    ("npm/package.json", r'("version":\s*")[\d]+\.[\d]+\.[\d]+[^"]*(")'),
    ("npm/package.json", r'("@sqlite-muninn/[^"]+": ")[\d]+\.[\d]+\.[\d]+[^"]*(")'),
]


# ======================================================================
# FILE DISCOVERY & DEPENDENCY SORTING
# ======================================================================

_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')

# Files excluded from main source/header discovery
_SOURCES_EXCLUDE = {"sqlite3_wasm_extra_init.c"}
_HEADERS_EXCLUDE = {"sqlite3.h", "sqlite3ext.h"}


def _discover(directory: str, pattern: str, exclude: set[str]) -> list[str]:
    """Discover files matching pattern in directory, minus exclusions."""
    return sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / directory).glob(pattern) if p.name not in exclude
    )


def _parse_internal_includes(filepath: Path, internal_names: set[str]) -> list[str]:
    """Parse #include "..." directives, returning only internal file references."""
    result = []
    for line in filepath.read_text().splitlines():
        m = _INCLUDE_RE.match(line)
        if m and m.group(1) in internal_names:
            result.append(m.group(1))
    return result


def _topo_sort_headers(files: list[str]) -> list[str]:
    """Topological sort header files by their #include dependencies (Kahn's algorithm).

    Headers that depend on nothing come first. Headers that #include other
    internal headers come after their dependencies. Ties broken alphabetically
    for deterministic output.
    """
    # Map bare filenames to full relative paths for include resolution
    name_to_path = {Path(f).name: f for f in files}
    internal_names = set(name_to_path)

    # Build dependency graph: file → set of files it depends on
    deps: dict[str, set[str]] = {}
    for f in files:
        includes = _parse_internal_includes(PROJECT_ROOT / f, internal_names)
        deps[f] = {name_to_path[inc] for inc in includes}

    # Kahn's algorithm
    in_degree = {f: len(deps[f]) for f in files}
    dependents: dict[str, list[str]] = {f: [] for f in files}
    for f, file_deps in deps.items():
        for dep in file_deps:
            dependents[dep].append(f)

    queue = sorted(f for f in files if in_degree[f] == 0)
    result = []

    while queue:
        f = queue.pop(0)
        result.append(f)
        for dep in sorted(dependents[f]):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
                queue.sort()

    if len(result) != len(files):
        missing = set(files) - set(result)
        raise ValueError(f"Circular #include dependency among: {missing}")

    return result


def _sort_sources_by_header_order(sources: list[str], sorted_headers: list[str]) -> list[str]:
    """Sort .c files by the position of their matching .h in the header order.

    Each .c file is matched to a .h file by stem (e.g. vec_math.c → vec_math.h).
    Files without a matching header sort to the end. This naturally puts
    muninn.c last since muninn.h depends on everything.
    """
    header_pos = {Path(h).stem: i for i, h in enumerate(sorted_headers)}

    def sort_key(src: str) -> tuple[int, str]:
        return (header_pos.get(Path(src).stem, len(sorted_headers)), src)

    return sorted(sources, key=sort_key)


# ── Discover and sort at import time ─────────────────────────────────
SOURCES_WASM_EXTRA = [f"src/{f}" for f in sorted(_SOURCES_EXCLUDE)]

HEADERS = _topo_sort_headers(_discover("src", "*.h", _HEADERS_EXCLUDE))
SOURCES = _sort_sources_by_header_order(_discover("src", "*.c", _SOURCES_EXCLUDE), HEADERS)
TEST_SOURCES = _discover("test", "test_*.c", set())

# WASM lite: exclude embed_gguf.c (needs llama.cpp)
_SOURCES_EXCLUDE_WASM_LITE = _SOURCES_EXCLUDE | {"embed_gguf.c"}
SOURCES_WASM_LITE = _sort_sources_by_header_order(_discover("src", "*.c", _SOURCES_EXCLUDE_WASM_LITE), HEADERS)

# ── Validation ───────────────────────────────────────────────────────
assert set(TEST_LINK_SOURCES) <= set(SOURCES), (
    f"TEST_LINK_SOURCES has files not in SOURCES: {set(TEST_LINK_SOURCES) - set(SOURCES)}"
)


# ======================================================================
# UTILITIES
# ======================================================================


def _read_version() -> str:
    return VERSION_FILE.read_text().strip()


def dirty(output_paths: list[Path] | Path, input_paths: list[Path] | Path) -> bool:
    """True if any output is missing or older than any input."""
    if isinstance(output_paths, Path):
        output_paths = [output_paths]
    if isinstance(input_paths, Path):
        input_paths = [input_paths]

    for out in output_paths:
        if not out.exists():
            return True

    newest_input = max(p.stat().st_mtime for p in input_paths if p.exists())
    oldest_output = min(p.stat().st_mtime for p in output_paths)
    return oldest_output < newest_input


def _cmake_flags_str(flags: dict[str, str]) -> str:
    """Format cmake flags as -DKEY=VAL space-separated string."""
    return " ".join(f"-D{k}={v}" for k, v in flags.items())


def _lib_subpath(lib_name: str) -> str:
    """Map library filename to its path under the cmake build/ directory."""
    if lib_name == "libllama.a":
        return "src/libllama.a"
    return f"ggml/src/{lib_name}"


def _llama_lib_paths() -> list[str]:
    """Full relative paths to llama.cpp static libraries."""
    return [f"vendor/llama.cpp/build/{_lib_subpath(lib)}" for lib in LLAMA_LIBS]


# ======================================================================
# QUERY DISPATCH
# ======================================================================

# Each entry: variable name → lambda returning its space-separated string value.
# Makefile evaluates these via: VAR := $(shell python3 scripts/generate_build.py query VAR)
QUERY_VARS: dict[str, callable] = {
    # File lists
    "SRC": lambda: " ".join(SOURCES),
    "HEADERS": lambda: " ".join(HEADERS),
    "TEST_SRC": lambda: " ".join(TEST_SOURCES),
    "TEST_LINK_SRC": lambda: " ".join(TEST_LINK_SOURCES),
    # WASM file lists (prefixed with ../ for wasm/ subdirectory)
    "MUNINN_SRC_WASM": lambda: " ".join(f"../{s}" for s in SOURCES),
    "MUNINN_SRC_WASM_LITE": lambda: " ".join(f"../{s}" for s in SOURCES_WASM_LITE),
    "SOURCES_WASM_EXTRA": lambda: " ".join(f"../{s}" for s in SOURCES_WASM_EXTRA),
    # WASM file lists (root-relative, for root Makefile integration)
    "MUNINN_SRC_WASM_LITE_ROOT": lambda: " ".join(SOURCES_WASM_LITE),
    "SOURCES_WASM_EXTRA_ROOT": lambda: " ".join(SOURCES_WASM_EXTRA),
    # CMake flags
    "LLAMA_CMAKE_FLAGS": lambda: _cmake_flags_str(CMAKE_FLAGS_BASE),
    "LLAMA_CMAKE_FLAGS_WASM": lambda: _cmake_flags_str(CMAKE_FLAGS_WASM),
    # llama.cpp paths
    "LLAMA_INCLUDE": lambda: " ".join(f"-I{d}" for d in LLAMA_INCLUDE_DIRS),
    "LLAMA_INCLUDE_WASM": lambda: " ".join(f"-I../{d}" for d in LLAMA_INCLUDE_DIRS),
    "LLAMA_LIBS_CORE": lambda: " ".join(_llama_lib_paths()),
}


def cmd_query(args: argparse.Namespace) -> int:
    var_name = args.var_name
    if var_name not in QUERY_VARS:
        log.error("Unknown variable: %s", var_name)
        log.error("Available: %s", ", ".join(sorted(QUERY_VARS)))
        return 1
    print(QUERY_VARS[var_name]())
    return 0


# ======================================================================
# SUBCOMMAND: windows
# ======================================================================

_WINDOWS_BAT_TEMPLATE = """\
@echo off
REM Build muninn.dll with MSVC + llama.cpp
REM Generated by scripts/generate_build.py — DO NOT EDIT
REM Requires: Visual Studio or Build Tools with MSVC, CMake
REM Usage: Run from "Developer Command Prompt for VS" or after ilammy/msvc-dev-cmd in CI

REM Step 1: Build llama.cpp static libraries via CMake
echo Building llama.cpp...
cmake -B vendor\\llama.cpp\\build -S vendor\\llama.cpp ^
{cmake_flags}

if %ERRORLEVEL% neq 0 (
    echo CMake configure failed
    exit /b %ERRORLEVEL%
)

cmake --build vendor\\llama.cpp\\build --config MinSizeRel -j %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo llama.cpp build failed
    exit /b %ERRORLEVEL%
)

echo llama.cpp built successfully

REM Step 2: Build muninn.dll linking against llama.cpp
if not exist build mkdir build

cl.exe /O2 /MT /W4 /LD /Isrc ^
{include_flags} ^
{source_files} ^
{lib_files} ^
    /Fe:build\\muninn.dll

if %ERRORLEVEL% neq 0 (
    echo Build failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Built build\\muninn.dll successfully
"""


def _bat_continuation(items: list[str]) -> str:
    """Join items with bat line-continuation (^ + newline)."""
    return " ^\n".join(items)


def cmd_windows(args: argparse.Namespace) -> int:
    out_dir = PROJECT_ROOT / "build" / "generated"
    out_file = out_dir / "build_windows.bat"
    input_paths = [PROJECT_ROOT / "scripts" / "generate_build.py"]

    if args.list_inputs:
        for p in input_paths:
            print(p.relative_to(PROJECT_ROOT))
        return 0

    if args.list_outputs:
        print(out_file.relative_to(PROJECT_ROOT))
        return 0

    if args.status:
        return 0 if not dirty(out_file, input_paths) else 1

    if not args.force and not dirty(out_file, input_paths):
        log.info("build/generated/build_windows.bat is up to date")
        return 0

    bs = "\\"  # backslash for f-strings
    content = _WINDOWS_BAT_TEMPLATE.format(
        cmake_flags=_bat_continuation(f"    -D{k}={v}" for k, v in CMAKE_FLAGS_BASE.items()),
        include_flags=_bat_continuation(f"    /I{d.replace('/', bs)}" for d in LLAMA_INCLUDE_DIRS),
        source_files=_bat_continuation(f"    {s.replace('/', bs)}" for s in SOURCES),
        lib_files=_bat_continuation(f"    {lib}" for lib in LLAMA_LIBS_WINDOWS),
    )

    if args.dry_run:
        log.info("DRY RUN: would write %s (%d bytes)", out_file.relative_to(PROJECT_ROOT), len(content))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content)
    log.info("Generated %s", out_file.relative_to(PROJECT_ROOT))
    return 0


# ======================================================================
# SUBCOMMAND: amalgamate
# ======================================================================


def _amalgamation_header(version: str, date: str) -> str:
    """Build the amalgamation file header from the data model."""
    include_flags = " ".join(f"-I{d}" for d in LLAMA_INCLUDE_DIRS)
    lib_lines = [f"vendor/llama.cpp/build/{_lib_subpath(lib)}" for lib in LLAMA_LIBS]
    lib_block = " \\\n *       ".join(lib_lines)

    return f"""\
/*
 * muninn amalgamation — v{version}
 * Generated {date}
 *
 * HNSW vector search + graph traversal + Node2Vec + GGUF embeddings for SQLite.
 * https://github.com/user/sqlite-muninn
 *
 * IMPORTANT: This amalgamation requires llama.cpp for GGUF embedding support.
 * You must provide the llama.cpp headers and link against its static libraries.
 *
 * Build as loadable extension (requires llama.cpp pre-built):
 *   # Linux
 *   cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DBUILD_SHARED_LIBS=OFF ...
 *   cmake --build vendor/llama.cpp/build
 *   gcc -O2 -fPIC -shared muninn.c -o muninn.so \\
 *       {include_flags} \\
 *       {lib_block} \\
 *       -lstdc++ -lm -lpthread
 *
 *   # macOS
 *   cc -O2 -fPIC -dynamiclib muninn.c -o muninn.dylib \\
 *       {include_flags} \\
 *       {lib_block} \\
 *       -lc++ -framework Accelerate -lm
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

"""


def cmd_amalgamate(args: argparse.Namespace) -> int:
    out_dir = PROJECT_ROOT / "dist"
    out_c = out_dir / "muninn.c"
    out_h = out_dir / "muninn.h"

    src_paths = [PROJECT_ROOT / s for s in SOURCES]
    hdr_paths = [PROJECT_ROOT / h for h in HEADERS]
    input_paths = src_paths + hdr_paths

    if args.list_inputs:
        for p in input_paths:
            print(p.relative_to(PROJECT_ROOT))
        return 0

    if args.list_outputs:
        print(out_c.relative_to(PROJECT_ROOT))
        print(out_h.relative_to(PROJECT_ROOT))
        return 0

    if args.status:
        return 0 if not dirty([out_c, out_h], input_paths) else 1

    if not args.force and not dirty([out_c, out_h], input_paths):
        log.info("dist/muninn.c and dist/muninn.h are up to date")
        return 0

    version = _read_version()
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append(_amalgamation_header(version, date_str))

    # Inline all internal headers (strip #include "..." lines)
    for hdr in HEADERS:
        hdr_path = PROJECT_ROOT / hdr
        lines.append(f"/* ──── {hdr} ──── */")
        for line in hdr_path.read_text().splitlines():
            if '#include "' in line:
                continue
            lines.append(line)
        lines.append("")

    # Inline all source files (strip #include "..." and SQLITE_EXTENSION_INIT1)
    for src in SOURCES:
        src_path = PROJECT_ROOT / src
        lines.append(f"/* ──── {src} ──── */")
        for line in src_path.read_text().splitlines():
            if '#include "' in line:
                continue
            if "SQLITE_EXTENSION_INIT1" in line:
                continue
            lines.append(line)
        lines.append("")

    content = "\n".join(lines)

    if args.dry_run:
        log.info("DRY RUN: would write dist/muninn.c (%d lines)", len(lines))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_c.write_text(content)

    # Copy public header
    shutil.copy2(PROJECT_ROOT / "src" / "muninn.h", out_h)

    log.info("Amalgamation: %s (%d lines)", out_c.relative_to(PROJECT_ROOT), len(lines))
    log.info("Header:       %s", out_h.relative_to(PROJECT_ROOT))
    return 0


# ======================================================================
# SUBCOMMAND: npm
# ======================================================================


def _read_npm_package() -> dict:
    return json.loads((PROJECT_ROOT / "npm" / "package.json").read_text())


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    log.info("wrote %s", path.relative_to(PROJECT_ROOT))


def _copy_readme(dest_dir: Path) -> None:
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, dest_dir / "README.md")
        log.info("copied README.md -> %s", (dest_dir / "README.md").relative_to(PROJECT_ROOT))


def cmd_npm(args: argparse.Namespace) -> int:
    dist_dir = PROJECT_ROOT / "dist" / "nodejs" / "platforms"
    npm_dir = PROJECT_ROOT / "npm"
    main_pkg_path = npm_dir / "package.json"

    input_paths = [main_pkg_path, PROJECT_ROOT / "README.md"]
    output_paths = [dist_dir / platform / "package.json" for platform in NPM_PLATFORMS]
    output_paths.append(dist_dir / "wasm" / "package.json")

    if args.list_inputs:
        for p in input_paths:
            print(p.relative_to(PROJECT_ROOT))
        return 0

    if args.list_outputs:
        for p in output_paths:
            print(p.relative_to(PROJECT_ROOT))
        return 0

    if args.status:
        return 0 if not dirty(output_paths, input_paths) else 1

    if args.dry_run:
        log.info("DRY RUN: would generate npm sub-packages")
        return 0

    log.info("Generating npm sub-packages from npm/package.json into dist/nodejs/platforms/...")
    main = _read_npm_package()
    scope = main["name"]

    # Platform binary packages
    for platform, (os_name, cpu, ext) in NPM_PLATFORMS.items():
        pkg_dir = dist_dir / platform
        pkg = {
            "name": f"@{scope}/{platform}",
            "version": main["version"],
            "description": f"{scope} native binary for {os_name} {cpu}",
            "repository": main["repository"],
            "license": main["license"],
            "os": [os_name],
            "cpu": [cpu],
            "files": [f"muninn.{ext}"],
        }
        _write_json(pkg_dir / "package.json", pkg)
        _copy_readme(pkg_dir)

    # WASM package
    wasm_dir = dist_dir / "wasm"
    wasm_pkg = {
        "name": f"@{scope}/wasm",
        "version": main["version"],
        "description": f"{scope} WebAssembly build - SQLite with HNSW, graph, and Node2Vec in the browser",
        "repository": main["repository"],
        "license": main["license"],
        "type": "module",
        "main": "muninn_sqlite3.js",
        "files": ["muninn_sqlite3.js", "muninn_sqlite3.wasm"],
    }
    _write_json(wasm_dir / "package.json", wasm_pkg)
    _copy_readme(wasm_dir)

    # Sync optionalDependencies in npm/package.json
    version = main["version"]
    optional = {f"@{scope}/{platform}": version for platform in NPM_PLATFORMS}
    if main.get("optionalDependencies") != optional:
        main["optionalDependencies"] = optional
        _write_json(main_pkg_path, main)
        log.info("updated optionalDependencies in npm/package.json")

    # Copy README to main npm package too
    _copy_readme(npm_dir)

    log.info("Done.")
    return 0


# ======================================================================
# SUBCOMMAND: version
# ======================================================================


def cmd_version(args: argparse.Namespace) -> int:
    version = _read_version()
    check_only = args.check
    all_ok = True

    for rel_path, pattern_str in VERSION_STAMP_TARGETS:
        replacement = rf"\g<1>{version}\2"
        pattern = re.compile(pattern_str)

        fpath = PROJECT_ROOT / rel_path
        if not fpath.exists():
            log.warning("skip: %s (not found)", rel_path)
            continue

        original = fpath.read_text()
        updated = pattern.sub(replacement, original)

        if original == updated:
            log.info("ok:   %s (already %s)", rel_path, version)
            continue

        if check_only:
            log.error("stale: %s (needs update to %s)", rel_path, version)
            all_ok = False
        else:
            fpath.write_text(updated)
            log.info("stamp: %s -> %s", rel_path, version)

    if not all_ok:
        log.error("Version mismatch detected. Run: python3 scripts/generate_build.py version")
        return 1

    if not check_only:
        log.info("Version %s stamped into all targets.", version)
    return 0


# ======================================================================
# CLI
# ======================================================================


def _add_file_gen_args(parser: argparse.ArgumentParser) -> None:
    """Add common flags for file-generating subcommands."""
    parser.add_argument("--status", action="store_true", help="Exit 0 if outputs up-to-date, 1 if dirty")
    parser.add_argument("--list-inputs", action="store_true", help="Print input file paths (one per line)")
    parser.add_argument("--list-outputs", action="store_true", help="Print output file paths (one per line)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if up-to-date")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Centralised build configuration for muninn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="Errors only")

    sub = parser.add_subparsers(dest="command", required=True)

    # query
    p_query = sub.add_parser("query", help="Print a build variable value")
    p_query.add_argument("var_name", help=f"Variable name ({', '.join(sorted(QUERY_VARS))})")

    # windows
    p_win = sub.add_parser("windows", help="Generate build/generated/build_windows.bat")
    _add_file_gen_args(p_win)

    # amalgamate
    p_amal = sub.add_parser("amalgamate", help="Generate dist/muninn.c + dist/muninn.h")
    _add_file_gen_args(p_amal)

    # npm
    p_npm = sub.add_parser("npm", help="Generate npm platform sub-packages")
    _add_file_gen_args(p_npm)

    # version
    p_ver = sub.add_parser("version", help="Stamp VERSION into target files")
    p_ver.add_argument("--check", action="store_true", help="Exit 1 if any file is out of date (don't modify)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    dispatch = {
        "query": cmd_query,
        "windows": cmd_windows,
        "amalgamate": cmd_amalgamate,
        "npm": cmd_npm,
        "version": cmd_version,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
