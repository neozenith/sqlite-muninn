# Muninn WASM  — Build
# All commands run from wasm/ directory (invoked via make -C wasm)

SQLITE_VERSION ?= 3510000
SQLITE_YEAR    ?= 2025

.PHONY: help build build-lite build-full dist

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

######################################################################
# BUILD
######################################################################

SQLITE_SRC   = build/sqlite3.c
WASM_JS      = build/muninn_sqlite3.js
WASM_BIN     = build/muninn_sqlite3.wasm


# llama.cpp WASM static libraries (build-full only)
LLAMA_DIR        = ../vendor/llama.cpp
LLAMA_WASM_BUILD = $(LLAMA_DIR)/build-wasm
LLAMA_WASM_LIBS  = $(LLAMA_WASM_BUILD)/src/libllama.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml-base.a \
                   $(LLAMA_WASM_BUILD)/ggml/src/libggml-cpu.a

# ── Single source of truth: scripts/generate_build.py ──
_Q := uv run ../scripts/generate_build.py query
LLAMA_INCLUDE_WASM     := $(shell $(_Q) LLAMA_INCLUDE_WASM)
MUNINN_SRC             := $(shell $(_Q) MUNINN_SRC_WASM)
MUNINN_SRC_LITE        := $(shell $(_Q) MUNINN_SRC_WASM_LITE)
SOURCES_WASM_EXTRA     := $(shell $(_Q) SOURCES_WASM_EXTRA)
LLAMA_CMAKE_FLAGS_WASM := $(shell $(_Q) LLAMA_CMAKE_FLAGS_WASM)

# Shared emcc flags for both lite and full builds
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
	-Ibuild -I../src

$(SQLITE_SRC):                                     ## Download SQLite amalgamation
	@mkdir -p build
	curl -sL "https://www.sqlite.org/$(SQLITE_YEAR)/sqlite-amalgamation-$(SQLITE_VERSION).zip" -o /tmp/sqlite_wasm.zip
	unzip -o /tmp/sqlite_wasm.zip -d /tmp/sqlite_wasm_amal
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.c build/sqlite3.c
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.h build/sqlite3.h
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3ext.h build/sqlite3ext.h
	rm -rf /tmp/sqlite_wasm.zip /tmp/sqlite_wasm_amal

$(LLAMA_WASM_LIBS): | $(LLAMA_DIR)/CMakeLists.txt  ## Build llama.cpp as WASM static libs
	@command -v emcmake >/dev/null 2>&1 || { echo "error: emcmake not found — install Emscripten SDK"; exit 1; }
	emcmake cmake -B $(LLAMA_WASM_BUILD) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS_WASM)
	emmake $(MAKE) -C $(LLAMA_WASM_BUILD) llama ggml -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── build-lite: no llama.cpp, no embed_gguf ──
build-wasm-lite: $(SQLITE_SRC)                          ## Build lite WASM (no llama.cpp/embeddings)
	@command -v emcc >/dev/null 2>&1 || { echo "error: emcc not found — install Emscripten SDK"; exit 1; }
	emcc $(EMCC_FLAGS) \
		-DMUNINN_NO_LLAMA \
		$(SQLITE_SRC) \
		$(MUNINN_SRC_LITE) \
		$(SOURCES_WASM_EXTRA) \
		-lm \
		-o $(WASM_JS)
	@echo "WASM lite build complete:"
	@ls -lh $(WASM_JS) $(WASM_BIN)

# ── build-full: includes llama.cpp + embed_gguf ──
build-wasm-full: $(SQLITE_SRC) $(LLAMA_WASM_LIBS)       ## Build full WASM (with llama.cpp/embeddings)
	@command -v emcc >/dev/null 2>&1 || { echo "error: emcc not found — install Emscripten SDK"; exit 1; }
	emcc $(EMCC_FLAGS) \
		$(LLAMA_INCLUDE_WASM) \
		$(SQLITE_SRC) \
		$(MUNINN_SRC) \
		$(SOURCES_WASM_EXTRA) \
		$(LLAMA_WASM_LIBS) \
		-lm \
		-o $(WASM_JS)
	@echo "WASM full build complete:"
	@ls -lh $(WASM_JS) $(WASM_BIN)

build: build-wasm-full                                  ## Build WASM module (full by default)

$(WASM_JS): build
$(WASM_BIN): build

######################################################################
# DIST
######################################################################

dist: $(WASM_JS) $(WASM_BIN)                                   ## Copy WASM artifacts to ../dist/
	@mkdir -p ../dist/nodejs/platforms/wasm/
	cp $(WASM_JS) ../dist/nodejs/platforms/wasm/
	cp $(WASM_BIN) ../dist/nodejs/platforms/wasm/
	@echo "WASM artifacts copied to dist/"
	@ls -lh ../dist/nodejs/platforms/wasm/muninn_sqlite3.*



clean:                                             ## Remove build artifacts
	rm -rf build/
	rm -rf $(LLAMA_WASM_BUILD)
	rm -f a.out.*
