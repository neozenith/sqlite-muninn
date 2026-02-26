######################################################################
# benchmarks/sessions_demo/Makefile — development tooling for sessions_demo
#   invoke via: make -C benchmarks/sessions_demo {target}
######################################################################

# Paths relative to project root (all commands run from project root via -C)
PKG = benchmarks/sessions_demo
PKG_ABS = $(shell cd ../.. && pwd)/benchmarks/sessions_demo
ROOT_ABS = $(shell cd ../.. && pwd)

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
GGUF_MODEL = ../../models/nomic-embed-text-v1.5.Q8_0.gguf

.PHONY: help format lint fix typecheck test coverage extension clean ci \
        build cache-init cache-update cache-rebuild cache-clear cache-status

help:                    ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Extension build (delegates to root Makefile) ──────────────────
$(EXTENSION):
	$(MAKE) -C ../.. all

extension: $(EXTENSION)  ## Build the muninn C extension

# ── CLI commands ─────────────────────────────────────────────────
build: $(EXTENSION) $(GGUF_MODEL) ## Run full build pipeline (ingest + chunks + embeddings)
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo build

cache-init:              ## Initialize SQLite cache schema
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo cache init

cache-update:            ## Incremental update from JSONL files
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo cache update

cache-rebuild:           ## Full cache rebuild (clear + update)
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo cache rebuild

cache-clear:             ## Clear all cached data
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo cache clear

cache-status:            ## Show cache status
	uv run --directory $(ROOT_ABS) python -m benchmarks.sessions_demo cache status

# ── Formatting ────────────────────────────────────────────────────
format:                  ## Format Python code with ruff + isort
	uvx ruff format $(PKG_ABS) --line-length 120
	uvx ruff check $(PKG_ABS) --line-length 120 --fix-only --select I

# ── Linting ───────────────────────────────────────────────────────
lint:                    ## Check Python code with ruff (no auto-fix)
	uvx ruff check $(PKG_ABS) --line-length 120 --statistics
	uvx ruff format $(PKG_ABS) --line-length 120 --check

# ── Auto-fix ──────────────────────────────────────────────────────
fix:                     ## Auto-fix lint issues + format
	uvx ruff check $(PKG_ABS) --line-length 120 --fix-only
	uvx ruff format $(PKG_ABS) --line-length 120

# ── Type checking ─────────────────────────────────────────────────
typecheck:               ## Run type checking with mypy
	uvx mypy $(PKG_ABS) --ignore-missing-imports

# ── Testing ───────────────────────────────────────────────────────
test: $(EXTENSION)       ## Run all sessions_demo tests
	uv run --directory $(ROOT_ABS) python -m pytest $(PKG_ABS)/tests/ -v --no-cov

# ── Coverage ──────────────────────────────────────────────────────
coverage: $(EXTENSION)   ## Run tests with coverage report
	uv run --directory $(ROOT_ABS) python -m pytest $(PKG_ABS)/tests/ -v \
		--cov=benchmarks.sessions_demo --cov-report=term-missing --cov-report=html:$(PKG_ABS)/htmlcov

# ── Cleanup ───────────────────────────────────────────────────────
clean:                   ## Remove __pycache__, .pyc, coverage artifacts, output
	find $(PKG_ABS) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(PKG_ABS)/htmlcov $(PKG_ABS)/.coverage
	rm -rf $(PKG_ABS)/output

# ── CI composite ──────────────────────────────────────────────────
ci: lint typecheck test  ## Run full CI pipeline (lint + typecheck + test)
