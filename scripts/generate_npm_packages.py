#!/usr/bin/env python3
"""Generate npm sub-package package.json files and copy README.md.

Reads npm/package.json as the single source of truth for version, license,
repository, and keywords. Generates into dist/nodejs/platforms/:
  - dist/nodejs/platforms/{platform}/package.json  (one per platform binary)
  - dist/nodejs/platforms/wasm/package.json
  - Copies root README.md next to each generated package.json

Also updates optionalDependencies in npm/package.json to match the version.

Usage:
    python scripts/generate_npm_packages.py
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NPM_DIR = ROOT / "npm"
DIST_DIR = ROOT / "dist" / "nodejs" / "platforms"
MAIN_PKG = NPM_DIR / "package.json"
README_SRC = ROOT / "README.md"

# Platform definitions: directory name -> (os, cpu, binary extension)
PLATFORMS = {
    "darwin-arm64": ("darwin", "arm64", "dylib"),
    "darwin-x64": ("darwin", "x64", "dylib"),
    "linux-x64": ("linux", "x64", "so"),
    "linux-arm64": ("linux", "arm64", "so"),
    "win32-x64": ("win32", "x64", "dll"),
}


def read_main_package() -> dict:
    return json.loads(MAIN_PKG.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    log.info("wrote %s", path.relative_to(ROOT))


def copy_readme(dest_dir: Path) -> None:
    if README_SRC.exists():
        shutil.copy2(README_SRC, dest_dir / "README.md")
        log.info("copied README.md -> %s", (dest_dir / "README.md").relative_to(ROOT))


def generate_platform_package(main: dict, platform: str, os_name: str, cpu: str, ext: str) -> None:
    pkg_dir = DIST_DIR / platform
    scope = main["name"]  # "sqlite-muninn" -> scope prefix "@sqlite-muninn"
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
    write_json(pkg_dir / "package.json", pkg)
    copy_readme(pkg_dir)


def generate_wasm_package(main: dict) -> None:
    pkg_dir = DIST_DIR / "wasm"
    scope = main["name"]
    pkg = {
        "name": f"@{scope}/wasm",
        "version": main["version"],
        "description": f"{scope} WebAssembly build - SQLite with HNSW, graph, and Node2Vec in the browser",
        "repository": main["repository"],
        "license": main["license"],
        "type": "module",
        "main": "muninn_sqlite3.js",
        "files": ["muninn_sqlite3.js", "muninn_sqlite3.wasm"],
    }
    write_json(pkg_dir / "package.json", pkg)
    copy_readme(pkg_dir)


def update_optional_dependencies(main: dict) -> None:
    """Ensure optionalDependencies in npm/package.json are version-synced."""
    version = main["version"]
    scope = main["name"]
    optional = {f"@{scope}/{platform}": version for platform in PLATFORMS}
    if main.get("optionalDependencies") != optional:
        main["optionalDependencies"] = optional
        write_json(MAIN_PKG, main)
        log.info("updated optionalDependencies in npm/package.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Generating npm sub-packages from npm/package.json into dist/nodejs/platforms/...")
    pkg = read_main_package()

    for platform, (os_name, cpu, ext) in PLATFORMS.items():
        generate_platform_package(pkg, platform, os_name, cpu, ext)

    generate_wasm_package(pkg)
    update_optional_dependencies(pkg)

    # Copy README to main npm package too
    copy_readme(NPM_DIR)

    log.info("Done.")


if __name__ == "__main__":
    main()
