#!/usr/bin/env python3
"""
E2E Colab Link Verification — Playwright-CLI

Opens each example's Colab badge URL in a real browser and verifies:
  1. Page loads (title contains notebook name)
  2. Notebook cells are visible (cell count > 0)
  3. No error page (no "Page not found" / "404")
  4. Screenshot captured for visual review

Usage:
  uv run examples/e2e_colab.py                    # test all examples
  uv run examples/e2e_colab.py semantic_search     # test one example
  uv run examples/e2e_colab.py --check-code        # also verify Colab detection code is present

Screenshots saved to examples/e2e_screenshots/
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

GITHUB_OWNER = "neozenith"
GITHUB_REPO = "sqlite-muninn"
GITHUB_BRANCH = "main"
COLAB_BASE = "https://colab.research.google.com/github"

EXAMPLES_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = EXAMPLES_DIR / "e2e_screenshots"


def discover_examples() -> list[str]:
    """Find all example directories that have a matching .py file."""
    return sorted(
        d.name
        for d in EXAMPLES_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.py").exists()
    )


def colab_url(name: str, *, cache_bust: bool = False) -> str:
    """Construct the Colab badge URL for an example."""
    url = f"{COLAB_BASE}/{GITHUB_OWNER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/examples/{name}/{name}.ipynb"
    if cache_bust:
        url += f"?_cb={int(time.time())}"
    return url


def run_cli(*args: str, timeout: int = 60) -> str:
    """Run a playwright-cli command and return stdout."""
    result = subprocess.run(
        ["playwright-cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        log.warning("playwright-cli %s failed: %s", args[0], result.stderr.strip())
    return result.stdout


def _parse_eval_result(stdout: str) -> str:
    """Extract the value from playwright-cli eval output.

    Format is:
        ### Result
        <value>
        ### Ran Playwright code
        ...
    """
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "### Result" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return ""


def check_title(name: str) -> bool:
    """Verify the page title contains the notebook filename."""
    stdout = run_cli("eval", "document.title")
    expected = f"{name}.ipynb"
    return expected in stdout


def check_cells_visible() -> bool:
    """Verify at least one notebook cell is rendered."""
    stdout = run_cli("eval", "document.querySelectorAll('[class*=\"cell\"]').length")
    value = _parse_eval_result(stdout)
    return value.isdigit() and int(value) > 0


def check_no_error() -> bool:
    """Verify no 404 / error page."""
    stdout = run_cli("eval", "document.title")
    error_signals = ["Page not found", "404", "Error", "not found"]
    return not any(sig.lower() in stdout.lower() for sig in error_signals)


def check_colab_code_present() -> bool:
    """Verify the Colab detection code is visible in notebook cells."""
    stdout = run_cli("eval", "document.body.innerText.includes('_IN_COLAB')")
    value = _parse_eval_result(stdout)
    return value.lower() == "true"


def test_example(name: str, *, check_code: bool = False) -> dict:
    """Run all checks for a single example. Returns results dict."""
    results = {
        "name": name,
        "url": colab_url(name),
        "title_ok": check_title(name),
        "cells_ok": check_cells_visible(),
        "no_error": check_no_error(),
    }
    if check_code:
        results["code_ok"] = check_colab_code_present()

    # Screenshot
    screenshot_path = SCREENSHOTS_DIR / f"{name}_colab.png"
    run_cli("screenshot", f"--filename={screenshot_path}")
    results["screenshot"] = str(screenshot_path)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse args
    check_code = "--check-code" in sys.argv
    cache_bust = "--cache-bust" in sys.argv
    filter_names = [a for a in sys.argv[1:] if not a.startswith("--")]

    all_examples = discover_examples()
    examples = filter_names if filter_names else all_examples

    # Validate filter
    for name in examples:
        if name not in all_examples:
            log.error("Unknown example: %s (available: %s)", name, all_examples)
            sys.exit(2)

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    log.info("Testing %d Colab links...", len(examples))

    # Clean up any existing browser sessions
    run_cli("close-all")

    # Open browser with first URL
    first_url = colab_url(examples[0], cache_bust=cache_bust)
    run_cli("open", first_url, timeout=90)

    results = []
    for i, name in enumerate(examples):
        if i > 0:
            run_cli("goto", colab_url(name, cache_bust=cache_bust), timeout=60)

        result = test_example(name, check_code=check_code)
        results.append(result)

        # Print result line
        checks = [
            ("title", result["title_ok"]),
            ("cells", result["cells_ok"]),
            ("no_err", result["no_error"]),
        ]
        if check_code:
            checks.append(("code", result.get("code_ok", False)))

        all_ok = all(v for _, v in checks)
        status = "PASS" if all_ok else "FAIL"
        detail = "  ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in checks)
        print(f"  {status}: {name:<25s}  {detail}")

    # Close browser
    run_cli("close")

    # Summary
    passed = sum(1 for r in results if r["title_ok"] and r["cells_ok"] and r["no_error"])
    print(f"\n{passed}/{len(results)} examples loaded successfully in Colab.")
    print(f"Screenshots: {SCREENSHOTS_DIR}/")

    if passed < len(results):
        print("\nFailed examples:")
        for r in results:
            if not (r["title_ok"] and r["cells_ok"] and r["no_error"]):
                print(f"  - {r['name']}: {r['url']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
