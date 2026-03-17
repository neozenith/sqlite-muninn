CC ?= cc
CFLAGS_BASE = -O2 -Wall -Wextra -Wpedantic -Werror -std=c11 -fPIC -D_POSIX_C_SOURCE=200809L
CFLAGS_EXTRA ?=
LDFLAGS = -lm

# Version from VERSION file
VERSION := $(shell cat VERSION 2>/dev/null || echo 0.0.0)

# ── Single source of truth: scripts/generate_build.py ──
_Q := uv run scripts/generate_build.py query

# Platform detection (from generate_build.py)
UNAME_S          := $(shell $(_Q) UNAME_S)
SHARED_FLAGS     := $(shell $(_Q) SHARED_FLAGS)
EXT              := $(shell $(_Q) EXT)
CFLAGS_BASE      += $(shell $(_Q) CFLAGS_PLATFORM)
LDFLAGS          += $(shell $(_Q) LDFLAGS_PLATFORM)

# llama.cpp vendored dependency
LLAMA_DIR     = vendor/llama.cpp
LLAMA_BUILD   = $(LLAMA_DIR)/build
LLAMA_INCLUDE    := $(shell $(_Q) LLAMA_INCLUDE)
LLAMA_LIBS_CORE  := $(shell $(_Q) LLAMA_LIBS_CORE)
LLAMA_CMAKE_FLAGS := $(shell $(_Q) LLAMA_CMAKE_FLAGS)

# Full library list: core + platform backends (blas, metal)
# On Linux, BLAS is optional — append via $(wildcard) if OpenBLAS was found.
ifeq ($(UNAME_S),Darwin)
    LLAMA_LIBS := $(shell $(_Q) LLAMA_LIBS)
else
    LLAMA_LIBS = $(LLAMA_LIBS_CORE) $(wildcard $(LLAMA_BUILD)/ggml/src/ggml-blas/libggml-blas.a)
endif

# macOS universal binary support: make ARCH=arm64 or make ARCH=x86_64
ifdef ARCH
    CFLAGS_BASE += -arch $(ARCH)
    LLAMA_CMAKE_FLAGS += -DCMAKE_OSX_ARCHITECTURES=$(ARCH)
endif

# SQLite for test linking
ifeq ($(UNAME_S),Darwin)
    SQLITE_PREFIX ?= $(shell brew --prefix sqlite 2>/dev/null || echo /usr/local)
    SQLITE_LIBS = -L$(SQLITE_PREFIX)/lib -lsqlite3
else ifeq ($(UNAME_S),Linux)
    SQLITE_LIBS ?= $(shell pkg-config --libs sqlite3 2>/dev/null || echo -lsqlite3)
else
    SQLITE_LIBS ?= -lsqlite3
endif

LDFLAGS_TEST = $(SQLITE_LIBS) $(LDFLAGS)

# ── WASM build configuration ──────────────────────────────────────────
SQLITE_VERSION  ?= 3510000
SQLITE_YEAR     ?= 2025
WASM_BUILD       = build/wasm
WASM_SQLITE_SRC  = $(WASM_BUILD)/sqlite3.c
WASM_JS          = $(WASM_BUILD)/muninn_sqlite3.js
WASM_BIN         = $(WASM_BUILD)/muninn_sqlite3.wasm

LLAMA_WASM_BUILD = $(LLAMA_DIR)/build-wasm
LLAMA_WASM_LIBS  = $(LLAMA_WASM_BUILD)/src/libllama.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml-base.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml-cpu.a

WASM_SRC_LITE        := $(shell $(_Q) MUNINN_SRC_WASM_LITE_ROOT)
WASM_SRC_EXTRA       := $(shell $(_Q) SOURCES_WASM_EXTRA_ROOT)
LLAMA_CMAKE_FLAGS_WASM := $(shell $(_Q) LLAMA_CMAKE_FLAGS_WASM)

EMCC_FLAGS = -O2 -msimd128 \
	-s WASM=1 \
	-s EXPORTED_FUNCTIONS='["_sqlite3_open","_sqlite3_close","_sqlite3_exec","_sqlite3_errmsg","_sqlite3_prepare_v2","_sqlite3_step","_sqlite3_finalize","_sqlite3_column_text","_sqlite3_column_int","_sqlite3_column_double","_sqlite3_column_blob","_sqlite3_column_bytes","_sqlite3_column_count","_sqlite3_column_name","_sqlite3_bind_text","_sqlite3_bind_int","_sqlite3_bind_double","_sqlite3_bind_blob","_sqlite3_bind_null","_sqlite3_reset","_sqlite3_free","_malloc","_free","_sqlite3_wasm_extra_init"]' \
	-s EXPORTED_RUNTIME_METHODS='["cwrap","ccall","UTF8ToString","stringToUTF8","getValue","setValue","FS","HEAPF32"]' \
	-s FORCE_FILESYSTEM=1 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=134217728 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME="createMuninnSQLite" \
	-DSQLITE_ENABLE_FTS5 \
	-DSQLITE_ENABLE_JSON1 \
	-DSQLITE_ENABLE_RTREE \
	-DSQLITE_THREADSAFE=0 \
	-DSQLITE_OMIT_LOAD_EXTENSION \
	-I$(WASM_BUILD) -Isrc

# Vendored C libraries (yyjson, etc.)
VENDOR_SRC     := $(shell $(_Q) VENDOR_SRC)
VENDOR_INCLUDE := $(shell $(_Q) VENDOR_INCLUDE)

# Source files — from generate_build.py
SRC          := $(shell $(_Q) SRC)
HEADERS      := $(shell $(_Q) HEADERS)
TEST_SRC     := $(shell $(_Q) TEST_SRC)
TEST_LINK_SRC := $(shell $(_Q) TEST_LINK_SRC)

.PHONY: all debug build test test-python test-js test-install test-all clean help \
        amalgamation install uninstall version version-stamp generate-windows \
        dist dist-extension dist-python dist-npm dist-wasm changelog release \
        docs-serve docs-build docs-clean \
        format format-c format-python format-js \
        lint lint-c lint-python lint-js \
        typecheck typecheck-python typecheck-js \
        ci ci-all llama-clean llamacpp \
        build-wasm build-wasm-lite build-wasm-full \
		llama-status llama-update

help:                                          ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

version:                                       ## Print version
	@echo $(VERSION)

######################################################################
# LLAMA.CPP PRE-BUILD
######################################################################
llama-status:
	# See what tag/commit your submodule currently points to                                                                                                                                                      
	git submodule status vendor/llama.cpp                                                                                                                                                                         

	# Fetch upstream without changing anything
	git -C vendor/llama.cpp fetch --tags

	# Compare current HEAD to latest upstream
	git -C vendor/llama.cpp log HEAD..origin/master --graph --format="%C(yellow)%h%C(reset) %C(cyan)%ad%C(reset) %C(auto)%d%C(reset) %s" --date=short

llama-update: llama-status
	# Update to latest commit on the tracked branch
	git submodule update --remote vendor/llama.cpp
	# OR
	# git -C vendor/llama.cpp checkout b8200   # or a full SHA, or a tag like v0.0.3

	# Then commit the pointer change in your repo
	git add vendor/llama.cpp
	git commit -m "chore: update llama.cpp submodule to latest"

llamacpp: $(LLAMA_LIBS_CORE)
$(LLAMA_LIBS_CORE): | $(LLAMA_DIR)/CMakeLists.txt
	@echo "######### Building llama.cpp static libraries (this may take a minute)..."
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS)

	@echo "######### Compiling llama.cpp 1 core..."
	cmake --build $(LLAMA_BUILD) --config MinSizeRel -j

llama-clean:                                   ## Clean llama.cpp build artifacts
	rm -rf $(LLAMA_BUILD)
	rm -rf $(LLAMA_WASM_BUILD)

######################################################################
# BUILD
######################################################################

all: build/muninn$(EXT)                        ## Build the extension

build: build/muninn$(EXT)
build/muninn$(EXT): $(SRC) $(VENDOR_SRC) $(LLAMA_LIBS)
	@mkdir -p build
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) $(SHARED_FLAGS) \
		-Isrc $(VENDOR_INCLUDE) $(LLAMA_INCLUDE) -o $@ $(SRC) $(VENDOR_SRC) $(LLAMA_LIBS) $(LDFLAGS)

debug: CFLAGS_BASE += -g -fsanitize=address,undefined -DDEBUG -O0
debug: LDFLAGS += -fsanitize=address,undefined
debug: build/muninn$(EXT)                      ## Build with ASan + UBSan

######################################################################
# WASM BUILD
######################################################################

$(WASM_SQLITE_SRC):                                ## Download SQLite amalgamation for WASM
	@mkdir -p $(WASM_BUILD)
	curl -sL "https://www.sqlite.org/$(SQLITE_YEAR)/sqlite-amalgamation-$(SQLITE_VERSION).zip" -o /tmp/sqlite_wasm.zip
	unzip -o /tmp/sqlite_wasm.zip -d /tmp/sqlite_wasm_amal
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.c $(WASM_BUILD)/sqlite3.c
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.h $(WASM_BUILD)/sqlite3.h
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3ext.h $(WASM_BUILD)/sqlite3ext.h
	rm -rf /tmp/sqlite_wasm.zip /tmp/sqlite_wasm_amal

$(LLAMA_WASM_LIBS): | $(LLAMA_DIR)/CMakeLists.txt ## Build llama.cpp as WASM static libs
	@command -v emcmake >/dev/null 2>&1 || { echo "error: emcmake not found — install Emscripten SDK"; exit 1; }
	emcmake cmake -B $(LLAMA_WASM_BUILD) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS_WASM)
	emmake $(MAKE) -C $(LLAMA_WASM_BUILD) llama ggml -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

build-wasm: build-wasm-full                        ## Build WASM module (lite by default)

build-wasm-lite: $(WASM_SQLITE_SRC)                ## Build lite WASM (no llama.cpp/embeddings)
	@command -v emcc >/dev/null 2>&1 || { echo "error: emcc not found — install Emscripten SDK"; exit 1; }
	emcc $(EMCC_FLAGS) \
		-DMUNINN_NO_LLAMA \
		$(VENDOR_INCLUDE) \
		$(WASM_SQLITE_SRC) \
		$(WASM_SRC_LITE) \
		$(VENDOR_SRC) \
		$(WASM_SRC_EXTRA) \
		-lm \
		-o $(WASM_JS)
	@echo "WASM lite build complete:"; ls -lh $(WASM_JS) $(WASM_BIN)

build-wasm-full: $(WASM_SQLITE_SRC) $(LLAMA_WASM_LIBS) ## Build full WASM (with llama.cpp/embeddings)
	@command -v emcc >/dev/null 2>&1 || { echo "error: emcc not found — install Emscripten SDK"; exit 1; }
	emcc $(EMCC_FLAGS) \
		$(VENDOR_INCLUDE) $(LLAMA_INCLUDE) \
		$(WASM_SQLITE_SRC) \
		$(SRC) \
		$(VENDOR_SRC) \
		$(WASM_SRC_EXTRA) \
		$(LLAMA_WASM_LIBS) \
		-lm \
		-o $(WASM_JS)
	@echo "WASM full build complete:"; ls -lh $(WASM_JS) $(WASM_BIN)

######################################################################
# TEST
######################################################################

test: build/test_runner                        ## Run C unit tests + coverage
	./build/test_runner
	@GCOVR=$$(command -v gcovr 2>/dev/null || echo .venv/bin/gcovr); \
	if [ -x "$$GCOVR" ]; then \
		$$GCOVR --root . --filter 'src/' --exclude 'src/sqlite3' \
			--gcov-ignore-errors=source_not_found \
			--gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file \
			--fail-under-line 50 --print-summary; \
	else \
		echo "gcovr not installed — skipping C coverage report"; \
	fi

build/test_runner: $(TEST_SRC) $(TEST_LINK_SRC) $(VENDOR_SRC) $(LLAMA_LIBS)
	@mkdir -p build
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) --coverage -DMUNINN_TESTING -Isrc $(VENDOR_INCLUDE) $(LLAMA_INCLUDE) -o $@ \
		$(TEST_SRC) $(TEST_LINK_SRC) $(VENDOR_SRC) $(LLAMA_LIBS) $(LDFLAGS_TEST)

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

init-python: .venv/.init-python                       ## Set up Python virtual environment and install dependencies
.venv/.init-python: pyproject.toml
	# Unfortunately llama.cpp's CMake hangs on a SVE check so we need to specify these flags here as well to get a working venv for the Python bindings.
	CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=ON" uv sync --all-groups
	@touch $@

format-python: .venv/.init-python                       ## Format Python code with ruff
	.venv/bin/ruff check --fix-only .
	.venv/bin/ruff format .

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

lint-python: .venv/.init-python                       ## Lint Python code with ruff
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

lint-js:                                       ## Lint TypeScript code with biome
	npm --prefix npm run lint

typecheck: typecheck-python typecheck-js format       ## Type-check all code

typecheck-python: .venv/.init-python                       ## Type-check Python with mypy
	.venv/bin/mypy sqlite_muninn/

typecheck-js:                                  ## Type-check TypeScript with tsc
	npm --prefix npm run typecheck

######################################################################
# PACKAGING
######################################################################

amalgamation: dist/muninn.c dist/muninn.h      ## Create single-file amalgamation

dist/muninn.c dist/muninn.h: $(SRC) $(HEADERS)
	uv run scripts/generate_build.py amalgamate

version-stamp:                                 ## Stamp VERSION into skill files + package.json
	uv run scripts/generate_build.py version
	npm --prefix ./npm install # update package-lock.json with new version

generate-windows:                              ## Generate build_windows.bat from centralised config
	uv run scripts/generate_build.py windows

examples-colab-jupytext:                       ## Generate Colab notebooks + enforce README badges
	uv run scripts/generate_build.py examples

examples-colab-check:                          ## Check notebooks + README badges are up to date
	uv run scripts/generate_build.py examples --status

dist: examples-colab-jupytext dist-extension dist-python dist-nodejs dist-wasm amalgamation changelog ## Build all distributable artifacts into dist/
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

dist-wasm: build-wasm-full                         ## Build WASM module into dist/ (requires emcc)
	@mkdir -p dist/nodejs/platforms/wasm/
	cp $(WASM_JS) dist/nodejs/platforms/wasm/
	cp $(WASM_BIN) dist/nodejs/platforms/wasm/
	@echo "WASM artifacts copied to dist/nodejs/platforms/wasm/"
	@ls -lh dist/nodejs/platforms/wasm/muninn_sqlite3.*

changelog: version-stamp                                     ## Generate CHANGELOG.md from git history
	.venv/bin/git-cliff -o CHANGELOG.md
	@echo "CHANGELOG.md updated"


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
	make -C docs/diagrams
	make -C benchmarks analyse-docs
	uv run mkdocs build --strict

docs-clean:                                    ## Clean documentation build
	rm -rf site/

######################################################################
# CI
######################################################################

ci: lint typecheck test test-python test-js docs-build    ## Full CI pipeline

ci-benchmarks-harness:                         ## CI for benchmarks/harness
	$(MAKE) -C benchmarks/harness ci

ci-benchmarks-demo-builder:                    ## CI for benchmarks/demo_builder
	$(MAKE) -C benchmarks/demo_builder ci

ci-benchmarks-sessions-demo:                   ## CI for benchmarks/sessions_demo
	$(MAKE) -C benchmarks/sessions_demo ci

# ci-viz:                                      ## CI for viz (disabled until kg-demo db restored)
# 	$(MAKE) -C viz ci

ci-all: ci ci-benchmarks-harness ci-benchmarks-demo-builder ci-benchmarks-sessions-demo  ## Full CI including subprojects

######################################################################
# CLEAN
######################################################################

viz-clean:
	$(MAKE) -C viz clean

clean: docs-clean llama-clean viz-clean          ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -f *.gcda *.gcno src/*.gcda src/*.gcno test/*.gcda test/*.gcno
	rm -rf .venv
	rm -rf a.out.*
	
