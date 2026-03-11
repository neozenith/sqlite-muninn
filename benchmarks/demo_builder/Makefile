######################################################################
# benchmarks/demo_builder/Makefile — development tooling for the demo builder
#   invoke via: make -C benchmarks/demo_builder {target}
######################################################################

# Paths relative to project root (all commands run from project root via -C)
PKG = benchmarks/demo_builder
PKG_ABS = $(shell cd ../.. && pwd)/benchmarks/demo_builder

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

.PHONY: help format lint fix typecheck test test-quick coverage extension clean ci \
        diagrams diagrams-clean

help:                    ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Extension build (delegates to root Makefile) ──────────────────
$(EXTENSION):
	$(MAKE) -C ../.. all

extension: $(EXTENSION)  ## Build the muninn C extension

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
test: $(EXTENSION)       ## Run all demo_builder tests
	uv run -m pytest $(PKG_ABS)/tests/ -v --no-cov

test-quick:              ## Run tests excluding slow/integration tests
	uv run -m pytest $(PKG_ABS)/tests/ -v -x -m "not slow" --no-cov

# ── Coverage ──────────────────────────────────────────────────────
coverage: $(EXTENSION)   ## Run tests with coverage report
	uv run -m pytest $(PKG_ABS)/tests/ -v \
		--cov=benchmarks.demo_builder --cov-report=term-missing --cov-report=html:$(PKG_ABS)/htmlcov

# ── Diagrams ─────────────────────────────────────────────────────
DIAGRAMS_DIR = diagrams
MMDC = npx -p @mermaid-js/mermaid-cli mmdc
MMDC_FLAGS = --scale 4 --backgroundColor white --outputFormat png

diagrams:                ## Render README Mermaid diagrams to PNG
	@mkdir -p $(DIAGRAMS_DIR)
	$(MMDC) -i README.md -a $(DIAGRAMS_DIR)/ $(MMDC_FLAGS)

diagrams-clean:          ## Remove rendered diagram artefacts
	rm -rf $(DIAGRAMS_DIR)

# ── Cleanup ───────────────────────────────────────────────────────
clean:                   ## Remove __pycache__, .pyc, coverage artifacts
	find $(PKG_ABS) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(PKG_ABS)/htmlcov $(PKG_ABS)/.coverage

# ── CI composite ──────────────────────────────────────────────────
ci: lint typecheck test diagrams  ## Run full CI pipeline (lint + typecheck + test)
