CC ?= cc
CFLAGS_BASE = -O2 -Wall -Wextra -Wpedantic -Werror -std=c11 -fPIC -D_POSIX_C_SOURCE=200809L
CFLAGS_EXTRA ?=
LDFLAGS = -lm

# Version from VERSION file
VERSION := $(shell cat VERSION 2>/dev/null || echo 0.0.0)

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED_FLAGS = -dynamiclib -undefined dynamic_lookup
    EXT = .dylib
    # macOS universal binary support: make ARCH=arm64 or make ARCH=x86_64
    ifdef ARCH
        CFLAGS_BASE += -arch $(ARCH)
    endif
    CFLAGS_BASE += -mmacosx-version-min=11.0
    # SQLite for test linking (extension only needs headers from src/)
    SQLITE_PREFIX ?= $(shell brew --prefix sqlite 2>/dev/null || echo /usr/local)
    SQLITE_LIBS = -L$(SQLITE_PREFIX)/lib -lsqlite3
else ifeq ($(UNAME_S),Linux)
    SHARED_FLAGS = -shared
    EXT = .so
    SQLITE_LIBS ?= $(shell pkg-config --libs sqlite3 2>/dev/null || echo -lsqlite3)
else
    SHARED_FLAGS = -shared
    EXT = .dll
    SQLITE_LIBS ?= -lsqlite3
endif

LDFLAGS_TEST = $(SQLITE_LIBS) $(LDFLAGS)

# Source files
SRC = src/muninn.c src/hnsw_vtab.c src/hnsw_algo.c \
      src/graph_tvf.c src/graph_load.c src/graph_centrality.c \
      src/graph_community.c src/node2vec.c src/vec_math.c \
      src/priority_queue.c src/id_validate.c

# Internal headers (excludes sqlite3.h / sqlite3ext.h)
HEADERS = src/vec_math.h src/priority_queue.h src/hnsw_algo.h \
          src/id_validate.h src/hnsw_vtab.h src/graph_common.h \
          src/graph_tvf.h src/graph_load.h src/graph_centrality.h \
          src/graph_community.h src/node2vec.h src/muninn.h

TEST_SRC = test/test_main.c test/test_vec_math.c test/test_priority_queue.c \
           test/test_hnsw_algo.c test/test_id_validate.c test/test_graph_load.c

.PHONY: all debug test test-python test-js test-install test-all clean help \
        amalgamation install uninstall version version-stamp \
        dist dist-extension dist-python dist-npm dist-wasm changelog release \
        docs-serve docs-build docs-wasm docs-clean \
        format format-c format-python format-js \
        lint lint-c lint-python lint-js \
        typecheck typecheck-python typecheck-js \
        ci ci-all

help:                                          ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

version:                                       ## Print version
	@echo $(VERSION)

######################################################################
# BUILD
######################################################################

all: build/muninn$(EXT)                        ## Build the extension

build/muninn$(EXT): $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) $(SHARED_FLAGS) -Isrc -o $@ $^ $(LDFLAGS)

debug: CFLAGS_BASE += -g -fsanitize=address,undefined -DDEBUG -O0
debug: LDFLAGS += -fsanitize=address,undefined
debug: build/muninn$(EXT)                      ## Build with ASan + UBSan

######################################################################
# TEST
######################################################################

test: build/test_runner                        ## Run C unit tests + coverage
	./build/test_runner
	@GCOVR=$$(command -v gcovr 2>/dev/null || echo .venv/bin/gcovr); \
	if [ -x "$$GCOVR" ]; then \
		$$GCOVR --root . --filter 'src/' --exclude 'src/sqlite3' \
			--fail-under-line 50 --print-summary; \
	else \
		echo "gcovr not installed — skipping C coverage report"; \
	fi

build/test_runner: $(TEST_SRC) src/vec_math.c src/priority_queue.c src/hnsw_algo.c src/id_validate.c src/graph_load.c
	@mkdir -p build
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) --coverage -Isrc -o $@ $^ $(LDFLAGS_TEST)

test-python: build/muninn$(EXT)                ## Run Python integration tests + coverage
	.venv/bin/python -m pytest pytests/ -v

test-js:                                       ## Run TypeScript tests + coverage
	npm --prefix npm test

test-all: test test-python test-js docs-build  ## Run all tests

######################################################################
# CODE QUALITY
######################################################################

format: format-c format-python format-js       ## Format all code

format-c:                                      ## Format C code with clang-format
	clang-format -i src/*.c src/*.h test/*.c test/*.h

format-python:                                 ## Format Python code with ruff
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix-only .

format-js:                                     ## Format TypeScript code with biome
	npm --prefix npm run format

lint: lint-c lint-python lint-js format               ## Lint all code

lint-c:                                        ## Lint C code with clang-format (check mode)
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format --dry-run --Werror src/*.c src/*.h test/*.c test/*.h 2>/dev/null; \
		echo "C lint passed"; \
	else \
		echo "clang-format not installed — skipping C lint"; \
	fi

lint-python:                                   ## Lint Python code with ruff
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

lint-js:                                       ## Lint TypeScript code with biome
	npm --prefix npm run lint

typecheck: typecheck-python typecheck-js format       ## Type-check all code

typecheck-python:                              ## Type-check Python with mypy
	.venv/bin/mypy sqlite_muninn/

typecheck-js:                                  ## Type-check TypeScript with tsc
	npm --prefix npm run typecheck

######################################################################
# PACKAGING
######################################################################

amalgamation: dist/muninn.c dist/muninn.h      ## Create single-file amalgamation

dist/muninn.c dist/muninn.h: $(SRC) $(HEADERS)
	bash scripts/amalgamate.sh

version-stamp:                                 ## Stamp VERSION into skill files + package.json
	.venv/bin/python scripts/version_stamp.py
	npm --prefix ./npm install # update package-lock.json with new version

dist: dist-extension dist-python dist-nodejs dist-wasm amalgamation changelog ## Build all distributable artifacts into dist/
	@echo ""
	@echo "All artifacts in dist/:"
	@ls -lh dist/ dist/python/ dist/nodejs/ 2>/dev/null
	@if [ -f dist/muninn_sqlite3.wasm ]; then echo ""; ls -lh dist/muninn_sqlite3.*; fi

dist-extension: version-stamp build/muninn$(EXT) ## Copy native extension to dist/
	@mkdir -p dist
	cp build/muninn$(EXT) dist/

dist-python: version-stamp build/muninn$(EXT)  ## Build Python wheel into dist/python/
	@mkdir -p dist/python
	uv build --wheel --out-dir dist/python

dist-nodejs: version-stamp                       ## Pack npm tarball into dist/nodejs/
	@mkdir -p dist/nodejs/
	npm pack --pack-destination dist/nodejs npm/

dist-wasm: amalgamation                       ## Build WASM module into dist/ (requires emcc)
	$(MAKE) -C wasm dist

changelog: version-stamp                                     ## Generate CHANGELOG.md from git history
	.venv/bin/git-cliff -o CHANGELOG.md
	@echo "CHANGELOG.md updated"

release:                                       ## Calculate next version from commits and prepare release
	$(eval NEW_VERSION := $(shell .venv/bin/git-cliff --bumped-version 2>/dev/null | sed 's/^v//'))
	@if [ -z "$(NEW_VERSION)" ]; then echo "error: could not determine next version"; exit 1; fi
	@echo "Next version: $(NEW_VERSION)"
	@echo "$(NEW_VERSION)" > VERSION
	$(MAKE) version-stamp
	.venv/bin/git-cliff --bump -o CHANGELOG.md
	@echo ""
	@echo "VERSION, CHANGELOG.md, and package manifests updated to $(NEW_VERSION)"
	@echo "Review changes, then: git add -A && git commit -m 'chore(release): $(NEW_VERSION)' && git tag v$(NEW_VERSION)"

######################################################################
# INSTALL
######################################################################

PREFIX ?= /usr/local

install: build/muninn$(EXT)                    ## Install extension and header
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 build/muninn$(EXT) $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/include
	install -m 644 src/muninn.h $(DESTDIR)$(PREFIX)/include/

uninstall:                                     ## Remove installed files
	rm -f $(DESTDIR)$(PREFIX)/lib/muninn$(EXT)
	rm -f $(DESTDIR)$(PREFIX)/include/muninn.h

test-install: build/muninn$(EXT)               ## Run install integration tests (pip + npm)
	.venv/bin/python -m pytest pytests/test_install.py -v -m integration --no-cov

######################################################################
# DOCUMENTATION
######################################################################

docs-serve: docs-build                         ## Serve docs locally with live reload
	uv run mkdocs serve

docs-build: version-stamp                      ## Build documentation site
	uv sync --all-groups
	make -C benchmarks analyze
	uv run mkdocs build --strict
	$(MAKE) docs-wasm

docs-wasm:                                     ## Copy WASM demo into built docs site
	@if [ -f wasm/build/muninn_sqlite3.js ] && [ -f wasm/assets/3300.db ]; then \
		echo "Copying WASM demo to site/examples/wasm/"; \
		mkdir -p site/examples/wasm/build site/examples/wasm/assets; \
		cp wasm/index.html site/examples/wasm/; \
		cp wasm/script.js site/examples/wasm/; \
		cp wasm/styles.css site/examples/wasm/; \
		cp wasm/build/muninn_sqlite3.js site/examples/wasm/build/; \
		cp wasm/build/muninn_sqlite3.wasm site/examples/wasm/build/; \
		cp wasm/assets/3300.db site/examples/wasm/assets/; \
		echo "WASM demo ready at site/examples/wasm/"; \
	else \
		echo "WASM demo artifacts not found — skipping (run 'make -C wasm build' first)"; \
	fi

docs-clean:                                    ## Clean documentation build
	rm -rf site/

######################################################################
# CI
######################################################################

ci: lint typecheck test test-python test-js docs-build    ## Full CI pipeline

ci-all: ci
	make -C viz ci
	make -C wasm ci

######################################################################
# CLEAN
######################################################################

clean: docs-clean                              ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -f *.gcda *.gcno src/*.gcda src/*.gcno test/*.gcda test/*.gcno
