######################################################################
# benchmarks/harness/Makefile — development tooling for the benchmark harness
#   invoke via: make -C benchmarks/harness {target}
######################################################################

# Paths relative to project root (all commands run from project root via -C)
HARNESS = benchmarks/harness
HARNESS_ABS = $(shell cd ../.. && pwd)/benchmarks/harness

# ── Platform detection for extension prerequisite ──────────────────
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    EXT = .dylib
else ifeq ($(UNAME_S),Linux)
    EXT = .so
else
    EXT = .dll
endif

EXTENSION = ../../build/muninn$(EXT)

.PHONY: help format lint fix typecheck test test-quick coverage extension clean

help:                    ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Extension build (delegates to root Makefile) ──────────────────
$(EXTENSION):
	$(MAKE) -C ../.. all

extension: $(EXTENSION)  ## Build the muninn C extension

# ── Formatting ────────────────────────────────────────────────────
format:                  ## Format Python code with ruff + isort
	uvx ruff format $(HARNESS_ABS) --line-length 120
	uvx ruff check $(HARNESS_ABS) --line-length 120 --fix-only --select I

# ── Linting ───────────────────────────────────────────────────────
lint:                    ## Check Python code with ruff (no auto-fix)
	uvx ruff check $(HARNESS_ABS) --line-length 120 --statistics
	uvx ruff format $(HARNESS_ABS) --line-length 120 --check

# ── Auto-fix ──────────────────────────────────────────────────────
fix:                     ## Auto-fix lint issues + format
	uvx ruff check $(HARNESS_ABS) --line-length 120 --fix-only
	uvx ruff format $(HARNESS_ABS) --line-length 120

# ── Type checking ─────────────────────────────────────────────────
typecheck:               ## Run type checking with mypy
	uvx mypy $(HARNESS_ABS) --ignore-missing-imports

# ── Testing ───────────────────────────────────────────────────────
test: $(EXTENSION)       ## Run all harness tests
	uv run -m pytest $(HARNESS_ABS)/tests/ -v

test-quick:              ## Run tests excluding slow/integration tests
	uv run -m pytest $(HARNESS_ABS)/tests/ -v -x -m "not slow"

# ── Coverage ──────────────────────────────────────────────────────
coverage: $(EXTENSION)   ## Run tests with coverage report
	uv run -m pytest $(HARNESS_ABS)/tests/ -v \
		--cov=benchmarks.harness --cov-report=term-missing --cov-report=html:$(HARNESS_ABS)/htmlcov

# ── Cleanup ───────────────────────────────────────────────────────
clean:                   ## Remove __pycache__, .pyc, coverage artifacts
	find $(HARNESS_ABS) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(HARNESS_ABS)/htmlcov $(HARNESS_ABS)/.coverage

# ── CI composite ──────────────────────────────────────────────────
ci: lint typecheck test  ## Run full CI pipeline (lint + typecheck + test)
