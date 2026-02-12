"""Integration tests for package installation.

These tests verify that `pip install` and `npm install` from the local repo
produce working packages. They create throwaway projects in temporary
directories and exercise the installed package entry points.

Marked with `pytest.mark.integration` — skipped by default during
`make test-python`. Run explicitly with:
    .venv/bin/python -m pytest pytests/test_install.py -v
"""

import pathlib
import shutil
import subprocess

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = (PROJECT_ROOT / "VERSION").read_text().strip()


def _run(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, raising on failure with full output."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        pytest.fail(f"Command failed: {' '.join(args)}\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return result


# ── pip install ──────────────────────────────────────────────


@pytest.mark.integration
class TestPipInstall:
    """Verify the Python package installs and works from a clean virtualenv."""

    def test_pip_install_version(self, tmp_path: pathlib.Path) -> None:
        """pip install from source should produce a package with the correct version."""
        # Copy the binary into the package so the wheel includes it
        ext_glob = list((PROJECT_ROOT / "build").glob("muninn.*"))
        ext_files = [f for f in ext_glob if f.suffix in (".so", ".dylib", ".dll")]
        if not ext_files:
            pytest.skip("Extension not built — run `make all` first")

        pkg_dir = PROJECT_ROOT / "sqlite_muninn"
        copied = pkg_dir / ext_files[0].name
        shutil.copy2(ext_files[0], copied)
        try:
            _run(["uv", "init", "--bare"], cwd=tmp_path)
            _run(["uv", "add", str(PROJECT_ROOT)], cwd=tmp_path)

            result = _run(
                ["uv", "run", "python", "-c", "import sqlite_muninn; print(sqlite_muninn.__version__)"],
                cwd=tmp_path,
            )
            assert result.stdout.strip() == VERSION

            result = _run(
                ["uv", "run", "python", "-c", "import sqlite_muninn; print(sqlite_muninn.loadable_path())"],
                cwd=tmp_path,
            )
            loadable = result.stdout.strip()
            assert "muninn" in loadable
        finally:
            copied.unlink(missing_ok=True)


# ── npm install ──────────────────────────────────────────────


@pytest.mark.integration
class TestNpmInstall:
    """Verify the npm package installs and works from a clean project."""

    def test_npm_install_loadable_path(self, tmp_path: pathlib.Path) -> None:
        """npm install from local should resolve the extension binary."""
        npm_dir = PROJECT_ROOT / "npm"
        if not (npm_dir / "dist" / "index.js").exists():
            pytest.skip("npm package not built — run `npm --prefix npm run build` first")

        ext_glob = list((PROJECT_ROOT / "build").glob("muninn.*"))
        ext_files = [f for f in ext_glob if f.suffix in (".so", ".dylib", ".dll")]
        if not ext_files:
            pytest.skip("Extension not built — run `make all` first")

        _run(["npm", "init", "-y"], cwd=tmp_path)
        _run(["npm", "install", str(npm_dir)], cwd=tmp_path)

        result = _run(
            ["node", "-e", "const m = require('sqlite-muninn'); console.log(m.getLoadablePath())"],
            cwd=tmp_path,
        )
        assert "muninn" in result.stdout.strip()

    def test_npm_install_version(self, tmp_path: pathlib.Path) -> None:
        """npm install should export the correct version."""
        npm_dir = PROJECT_ROOT / "npm"
        if not (npm_dir / "dist" / "index.js").exists():
            pytest.skip("npm package not built — run `npm --prefix npm run build` first")

        _run(["npm", "init", "-y"], cwd=tmp_path)
        _run(["npm", "install", str(npm_dir)], cwd=tmp_path)

        result = _run(
            ["node", "-e", "const m = require('sqlite-muninn'); console.log(m.version)"],
            cwd=tmp_path,
        )
        assert result.stdout.strip() == VERSION
