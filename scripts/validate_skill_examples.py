"""
Validate code examples in SKILL.md and reference cookbooks.

Recursively finds all .md files under the given directory, extracts fenced
code blocks (```sql, ```python), and executes them against the built extension.

Usage:
    python scripts/validate_skill_examples.py skills/muninn/

Exit code 0 if all examples pass, 1 if any fail.
"""
import logging
import re
import sqlite3
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXTENSION_PATH = PROJECT_ROOT / "muninn"

# Regex to match fenced code blocks: ```lang\n...\n```
CODE_BLOCK_RE = re.compile(
    r"```(\w+)\n(.*?)```",
    re.DOTALL,
)

# SQL blocks that are informational only (contain placeholders)
SQL_SKIP_PATTERNS = [
    "MATCH ?",          # Requires blob parameter
    "VALUES (?, ?)",    # Requires bound parameters
    "WHERE vector",     # Requires blob parameter
    "node2vec_train(",  # Needs populated edge table
    "zeroblob",         # Example placeholder
    "DELETE FROM",      # DML depends on prior CREATE context
]

# Python blocks that require external deps
PYTHON_SKIP_PATTERNS = [
    "import sqlite_muninn",  # Package not installed during validation
    "from sklearn",
    "import numpy",
    "import express",
    "import Database from",
    "from dotenv",
    "from sentence_transformers",
]


def load_extension(db: sqlite3.Connection) -> None:
    """Load the muninn extension into a connection."""
    db.enable_load_extension(True)
    db.load_extension(str(EXTENSION_PATH))
    db.enable_load_extension(False)


def extract_code_blocks(md_path: Path) -> list[tuple[str, str, int]]:
    """Extract (language, code, line_number) tuples from a markdown file."""
    content = md_path.read_text(encoding="utf-8")
    blocks = []
    for match in CODE_BLOCK_RE.finditer(content):
        lang = match.group(1).lower()
        code = match.group(2)
        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((lang, code.strip(), line_num))
    return blocks


def should_skip_sql(code: str) -> bool:
    """Check if SQL block should be skipped (requires parameters)."""
    return any(pattern in code for pattern in SQL_SKIP_PATTERNS)


def should_skip_python(code: str) -> bool:
    """Check if Python block should be skipped (requires external deps)."""
    return any(pattern in code for pattern in PYTHON_SKIP_PATTERNS)


def validate_sql_block(code: str, md_path: Path, line_num: int) -> bool:
    """Execute a SQL block against a fresh connection with muninn loaded."""
    if should_skip_sql(code):
        log.debug("  SKIP (requires parameters): %s:%d", md_path.name, line_num)
        return True

    try:
        db = sqlite3.connect(":memory:")
        load_extension(db)

        # Create common test fixtures
        db.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        db.execute("INSERT INTO edges VALUES ('a', 'b', 1.0)")
        db.execute("INSERT INTO edges VALUES ('b', 'c', 1.0)")
        db.execute("INSERT INTO edges VALUES ('c', 'a', 1.0)")
        db.commit()

        # Execute each statement
        for stmt in code.split(";"):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--") or stmt.startswith("."):
                continue
            # Skip statements with parameter placeholders
            if "?" in stmt:
                continue
            db.execute(stmt)

        db.close()
        return True
    except Exception as e:
        log.error("  FAIL SQL %s:%d — %s", md_path.name, line_num, e)
        log.error("  Code: %s", textwrap.shorten(code, 120))
        return False


def validate_python_block(code: str, md_path: Path, line_num: int) -> bool:
    """Check if a Python block parses correctly (syntax validation only)."""
    if should_skip_python(code):
        log.debug("  SKIP (external deps): %s:%d", md_path.name, line_num)
        return True

    try:
        compile(code, f"{md_path}:{line_num}", "exec")
        return True
    except SyntaxError as e:
        log.error("  FAIL Python syntax %s:%d — %s", md_path.name, line_num, e)
        return False


def validate_directory(skills_dir: Path) -> tuple[int, int, int]:
    """Validate all markdown files in the directory. Returns (passed, failed, skipped)."""
    passed = 0
    failed = 0
    skipped = 0

    md_files = sorted(skills_dir.rglob("*.md"))
    if not md_files:
        log.warning("No .md files found in %s", skills_dir)
        return 0, 0, 0

    for md_path in md_files:
        log.info("Validating %s", md_path.relative_to(skills_dir.parent.parent))
        blocks = extract_code_blocks(md_path)

        for lang, code, line_num in blocks:
            if lang == "sql":
                if should_skip_sql(code):
                    skipped += 1
                elif validate_sql_block(code, md_path, line_num):
                    passed += 1
                else:
                    failed += 1
            elif lang == "python":
                if should_skip_python(code):
                    skipped += 1
                elif validate_python_block(code, md_path, line_num):
                    passed += 1
                else:
                    failed += 1
            elif lang in ("javascript", "js", "typescript", "ts"):
                # JS/TS: syntax check only (would need Node.js)
                skipped += 1
            elif lang in ("c", "bash", "bat", "ruby", "cmake", "toml", "json"):
                skipped += 1
            else:
                skipped += 1

    return passed, failed, skipped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        log.error("Usage: python %s <skills_directory>", sys.argv[0])
        return 1

    skills_dir = Path(sys.argv[1])
    if not skills_dir.is_dir():
        log.error("Not a directory: %s", skills_dir)
        return 1

    # Check extension exists
    ext_glob = list(PROJECT_ROOT.glob("muninn.*"))
    if not ext_glob:
        log.error("Extension not built. Run: make all")
        return 1

    log.info("Validating skill examples in %s", skills_dir)
    log.info("")

    passed, failed, skipped = validate_directory(skills_dir)

    log.info("")
    log.info("Results: %d passed, %d failed, %d skipped", passed, failed, skipped)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
