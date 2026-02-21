# ruff: noqa: E501
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Export environment variables from a .env file as shell export/unset statements.

Use in combination with `eval` to load environment variables into the current shell:
    eval "$(uv run scripts/exportenv.py)"

Or to unset all variables defined in the .env file:
    eval "$(uv run scripts/exportenv.py --unset)"
"""
import argparse
import logging
import shlex
from pathlib import Path
from textwrap import dedent

# Script configuration
SCRIPT = Path(__file__)
SCRIPT_NAME = SCRIPT.stem
SCRIPT_DIR = SCRIPT.parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

log = logging.getLogger(__name__)


def _parse_env_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single .env line into a key-value pair.

    Handles quoted values and inline comments.
    Returns (None, None) for empty lines, comments, or invalid lines.
    """
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None, None

    key, value = line.split("=", 1)
    key = key.strip()

    # Use shlex to process the value (handles quotes and comments)
    lexer = shlex.shlex(value, posix=True)
    lexer.whitespace_split = True
    value = "".join(lexer)

    return key, value


def read_env_file(file_path: str | Path) -> dict[str, str] | None:
    """Read a .env file and return a dictionary of key-value pairs.

    Args:
        file_path: Path to the .env file

    Returns:
        Dictionary of environment variables, or None if file doesn't exist
    """
    file = file_path if isinstance(file_path, Path) else Path(file_path)
    if not file.is_file():
        return None

    return {
        key: value
        for key, value in map(_parse_env_line, file.read_text().splitlines())
        if key is not None and value is not None
    }


def main(unset: bool = False, env_dir: Path = Path.cwd()) -> int:
    """Export or unset environment variables from a .env file.

    Args:
        unset: If True, print unset statements instead of export
        env_dir: Directory containing the .env file

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    env_file = env_dir / ".env"

    if not env_file.exists():
        log.error(f"No .env file found at: {env_file}")
        return 1

    env_values = read_env_file(env_file)
    if not env_values:
        log.warning(f"No environment variables found in: {env_file}")
        return 0

    log.debug(f"Found {len(env_values)} environment variables in {env_file}")

    for key, value in env_values.items():
        if unset:
            print(f"unset {key}")
        else:
            # Quote the value to handle special characters
            print(f"export {key}={shlex.quote(value)}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent(f"""\
        {SCRIPT_NAME} - Export environment variables from a .env file.

        Reads a .env file and outputs shell-compatible export or unset statements.
        Designed to be used with `eval` for loading into the current shell session.

        INPUTS:
            .env file in the specified directory (default: current working directory)

        OUTPUTS:
            Shell export/unset statements to stdout

        EXAMPLES:
            # Load .env into current shell
            eval "$(uv run scripts/{SCRIPT_NAME}.py)"

            # Load from a specific directory
            eval "$(uv run scripts/{SCRIPT_NAME}.py --dir /path/to/project)"

            # Unset all variables defined in .env
            eval "$(uv run scripts/{SCRIPT_NAME}.py --unset)"

            # Debug mode to see what's happening
            uv run scripts/{SCRIPT_NAME}.py --verbose
        """),
        epilog=dedent("""\
        NOTES:
            - Handles quoted values and inline comments in .env files
            - Values are automatically shell-quoted in output for safety
            - Comments (lines starting with #) are ignored
            - Empty lines are ignored
        """),
    )

    parser.add_argument(
        "-d", "--dir",
        type=Path,
        default=Path.cwd(),
        metavar="DIR",
        help="Directory containing the .env file (default: current working directory)",
    )
    parser.add_argument(
        "-u", "--unset",
        action="store_true",
        help="Print unset statements instead of export statements",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    args = parser.parse_args()

    # Configure logging based on verbosity flags
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO,
        format="# %(levelname)s: %(message)s" if args.verbose else "# %(message)s",
    )

    log.debug(f"Working directory: {Path.cwd()}")
    log.debug(f"Looking for .env in: {args.dir}")

    exit_code = main(unset=args.unset, env_dir=args.dir)
    raise SystemExit(exit_code)
