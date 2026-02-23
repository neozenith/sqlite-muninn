# Muninn WASM Demo — Build, serve, and test targets
# All commands run from wasm/ directory (invoked via make -C wasm)

SQLITE_VERSION ?= 3510000
SQLITE_YEAR    ?= 2025
PORT           ?= 8300

.PHONY: help build build-lite build-full dev dist test e2e-videos format clean

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

######################################################################
# BUILD
######################################################################

SQLITE_SRC   = build/sqlite3.c
WASM_JS      = build/muninn_sqlite3.js
WASM_BIN     = build/muninn_sqlite3.wasm
DEMO_DB	  	 = assets/3300.db
DEMO_DB_BUILD = ../benchmarks/kg/3300.db

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
build-lite: $(SQLITE_SRC)                          ## Build lite WASM (no llama.cpp/embeddings)
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
build-full: $(SQLITE_SRC) $(LLAMA_WASM_LIBS)       ## Build full WASM (with llama.cpp/embeddings)
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

build: build-lite                                  ## Build WASM module (lite by default)

######################################################################
# DEV
######################################################################
$(DEMO_DB_BUILD):
	make -C ../benchmarks kg-demo

$(DEMO_DB): $(DEMO_DB_BUILD)   ## Download demo DB if missing
	@mkdir -p $(dir $@)
	cp $< $@
	@echo "Demo DB ready at $@"

dev: build $(DEMO_DB)                                         ## Start dev server on port 8300
	@echo "Starting dev server at http://localhost:$(PORT)"
	python3 -m http.server $(PORT) --directory .

######################################################################
# DIST
######################################################################

dist: $(WASM_JS)                                   ## Copy WASM artifacts to ../dist/
	@mkdir -p ../dist/nodejs/platforms/wasm/
	cp build/muninn_sqlite3.js ../dist/nodejs/platforms/wasm/
	cp build/muninn_sqlite3.wasm ../dist/nodejs/platforms/wasm/
	@echo "WASM artifacts copied to dist/"
	@ls -lh ../dist/nodejs/platforms/wasm/muninn_sqlite3.*

######################################################################
# TEST
######################################################################

test: build                                        ## Run Playwright E2E tests
	npm install
	npx playwright install chromium
	npx playwright test

######################################################################
# VIDEO CONVERSION
######################################################################

E2E_DIR = test-results

%.mp4: %.webm                                      ## Convert .webm to .mp4 (2x speed, 10fps)
	ffmpeg -i $< -vf "setpts=0.5*PTS,fps=10" -s 1280x720 -c:v libx264 -crf 20 -an -y $@

%.gif: %.webm                                      ## Convert .webm to .gif (2x speed, 10fps)
	ffmpeg -i $< -vf "setpts=0.5*PTS,fps=10,scale=1280:-1:flags=lanczos,palettegen" -y $@.palette.png
	ffmpeg -i $< -i $@.palette.png -filter_complex "setpts=0.5*PTS,fps=10,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse" -y $@
	rm -f $@.palette.png

e2e-videos: test                                   ## Convert all E2E videos
	$(MAKE) $(patsubst %.webm,%.mp4,$(wildcard $(E2E_DIR)/**/*.webm))
	$(MAKE) $(patsubst %.webm,%.gif,$(wildcard $(E2E_DIR)/**/*.webm))

ci: format test e2e-videos                               ## Run tests + convert E2E videos
######################################################################
# FORMAT
######################################################################

format:                                            ## Format JS/HTML/CSS with prettier
	npx prettier --write '*.{html,js,css}'

######################################################################
# CLEAN
######################################################################

clean:                                             ## Remove build artifacts
	rm -rf build/ test-results/ screenshots/ node_modules/
	rm -rf $(LLAMA_WASM_BUILD)
	rm -f a.out.*
