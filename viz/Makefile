# muninn-viz — Build & Dev targets
# All commands run from viz/ directory (invoked via make -C viz)

BACKEND_PORT ?= 8200
FRONTEND_PORT ?= 5282

.PHONY: help install install-python install-frontend dev test test-api test-frontend test-e2e \
       fix fix-python fix-frontend lint lint-python lint-frontend \
       typecheck typecheck-python typecheck-frontend ci clean

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: install-python install-frontend           ## Install all dependencies

install-python:                                    ## Install Python deps
	uv sync --directory .

install-frontend:                                  ## Install frontend deps
	npm --prefix frontend install

dev: install                                       ## Start dev servers (backend + frontend)
	@echo "http://localhost:$(FRONTEND_PORT)/"
	npx --prefix frontend concurrently --kill-others \
		--names "backend,frontend" --prefix-colors "blue,green" \
		"uv run --directory . python -m server --port $(BACKEND_PORT) --reload" \
		"VITE_API_PORT=$(BACKEND_PORT) npm --prefix frontend run dev -- --port $(FRONTEND_PORT)"

# ── Tests ─────────────────────────────────────────────────────────────

test: test-api test-frontend                       ## Run unit tests (API + frontend)

test-api: install-python                           ## Python API tests
	uv run --directory . pytest -v

test-frontend: install-frontend                    ## TypeScript unit tests (vitest)
	npm --prefix frontend run test

test-e2e: install                                  ## Playwright E2E tests
	npm --prefix frontend run test:e2e

ci: lint typecheck test test-e2e                   ## Full CI: lint + typecheck + test + e2e

# ── Code Quality ─────────────────────────────────────────────────────

fix: fix-python fix-frontend                       ## Format all code

fix-python:                                        ## Format Python code with ruff
	uv run --directory . ruff format .
	uv run --directory . ruff check --fix-only .

fix-frontend: install-frontend                     ## Format TypeScript code with prettier
	npm --prefix frontend run format

lint: lint-python lint-frontend                    ## Lint all code

lint-python:                                       ## Lint Python code with ruff
	uv run --directory . ruff check .
	uv run --directory . ruff format --check .

lint-frontend: install-frontend                    ## Lint TypeScript code with eslint + prettier
	npm --prefix frontend run lint
	npm --prefix frontend run format:check

typecheck: typecheck-python typecheck-frontend     ## Type-check all code

typecheck-python:                                  ## Type-check Python with mypy
	uv run --directory . mypy server/

typecheck-frontend: install-frontend               ## Type-check TypeScript with tsc
	npm --prefix frontend run typecheck

# ── Clean ─────────────────────────────────────────────────────────────

clean:                                             ## Remove build artifacts
	rm -rf frontend/dist frontend/node_modules/.cache
	rm -rf frontend/test-results frontend/coverage
	rm -rf .pytest_cache __pycache__ server/__pycache__ tests/__pycache__
	rm -rf .coverage
