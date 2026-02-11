"""Custom build that compiles the muninn C extension and copies it into the Python package."""
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

EXT_MAP = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}


class BuildMuninn(build_py):
    """Build the C extension before collecting Python packages."""

    def run(self):
        ext = EXT_MAP.get(platform.system(), ".so")
        binary = Path(f"muninn{ext}")
        pkg_dir = Path("sqlite_muninn")

        # Build if binary doesn't exist at the repo root
        if not binary.exists():
            try:
                subprocess.check_call(["make", "all"])
            except (subprocess.CalledProcessError, FileNotFoundError):
                warnings.warn(
                    "Could not build muninn C extension. "
                    "Install from PyPI for pre-built binaries: pip install sqlite-muninn",
                    stacklevel=2,
                )

        # Copy binary into the package directory for inclusion in the wheel
        if binary.exists():
            shutil.copy2(binary, pkg_dir / binary.name)

        super().run()


setup(cmdclass={"build_py": BuildMuninn})
