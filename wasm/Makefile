# Muninn WASM Demo — Build, serve, and test targets
# All commands run from wasm/ directory (invoked via make -C wasm)

SQLITE_VERSION ?= 3510000
SQLITE_YEAR    ?= 2025
PORT           ?= 8300

.PHONY: help build dev dist test e2e-videos format clean

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

######################################################################
# BUILD
######################################################################

AMALGAMATION = ../dist/muninn.c
SQLITE_SRC   = build/sqlite3.c
WASM_JS      = build/muninn_sqlite3.js
WASM_BIN     = build/muninn_sqlite3.wasm
DEMO_DB	  	 = assets/3300.db
DEMO_DB_BUILD = ../benchmarks/kg/3300.db

$(AMALGAMATION):
	$(MAKE) -C .. amalgamation

$(SQLITE_SRC):                                     ## Download SQLite amalgamation
	@mkdir -p build
	curl -sL "https://www.sqlite.org/$(SQLITE_YEAR)/sqlite-amalgamation-$(SQLITE_VERSION).zip" -o /tmp/sqlite_wasm.zip
	unzip -o /tmp/sqlite_wasm.zip -d /tmp/sqlite_wasm_amal
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.c build/sqlite3.c
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3.h build/sqlite3.h
	cp /tmp/sqlite_wasm_amal/sqlite-amalgamation-*/sqlite3ext.h build/sqlite3ext.h
	rm -rf /tmp/sqlite_wasm.zip /tmp/sqlite_wasm_amal

$(WASM_JS): $(AMALGAMATION) $(SQLITE_SRC)          ## Build WASM artifacts
	@command -v emcc >/dev/null 2>&1 || { echo "error: emcc not found — install Emscripten SDK"; exit 1; }
	emcc -O2 \
		-s WASM=1 \
		-s EXPORTED_FUNCTIONS='["_sqlite3_open","_sqlite3_close","_sqlite3_exec","_sqlite3_errmsg","_sqlite3_prepare_v2","_sqlite3_step","_sqlite3_finalize","_sqlite3_column_text","_sqlite3_column_int","_sqlite3_column_double","_sqlite3_column_blob","_sqlite3_column_bytes","_sqlite3_column_count","_sqlite3_column_name","_sqlite3_bind_text","_sqlite3_bind_int","_sqlite3_bind_double","_sqlite3_bind_blob","_sqlite3_bind_null","_sqlite3_reset","_sqlite3_free","_malloc","_free","_sqlite3_wasm_extra_init"]' \
		-s EXPORTED_RUNTIME_METHODS='["cwrap","ccall","UTF8ToString","stringToUTF8","getValue","setValue","FS","HEAPF32"]' \
		-s FORCE_FILESYSTEM=1 \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s INITIAL_MEMORY=16777216 \
		-s MODULARIZE=1 \
		-s EXPORT_NAME="createMuninnSQLite" \
		-DSQLITE_ENABLE_FTS5 \
		-DSQLITE_ENABLE_JSON1 \
		-DSQLITE_ENABLE_RTREE \
		-DSQLITE_THREADSAFE=0 \
		-DSQLITE_OMIT_LOAD_EXTENSION \
		-Ibuild \
		$(SQLITE_SRC) \
		$(AMALGAMATION) \
		../src/sqlite3_wasm_extra_init.c \
		-lm \
		-o $(WASM_JS)
	@echo "WASM build complete:"
	@ls -lh $(WASM_JS) $(WASM_BIN)



build: $(WASM_JS)                                  ## Build WASM module to build/

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
	@mkdir -p ../dist
	cp build/muninn_sqlite3.js ../dist/
	cp build/muninn_sqlite3.wasm ../dist/
	@echo "WASM artifacts copied to dist/"
	@ls -lh ../dist/muninn_sqlite3.*

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
