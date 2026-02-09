CC ?= cc
CFLAGS_BASE = -O2 -Wall -Wextra -Wpedantic -std=c11 -fPIC
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED_FLAGS = -dynamiclib -undefined dynamic_lookup
    EXT = .dylib
    # Use Homebrew SQLite on macOS (system SQLite may lack extension support)
    SQLITE_PREFIX ?= $(shell brew --prefix sqlite 2>/dev/null || echo /usr/local)
    CFLAGS_BASE += -I$(SQLITE_PREFIX)/include
    LDFLAGS_TEST = -L$(SQLITE_PREFIX)/lib -lsqlite3 $(LDFLAGS)
else ifeq ($(UNAME_S),Linux)
    SHARED_FLAGS = -shared
    EXT = .so
    LDFLAGS_TEST = -lsqlite3 $(LDFLAGS)
else
    SHARED_FLAGS = -shared
    EXT = .dll
    LDFLAGS_TEST = -lsqlite3 $(LDFLAGS)
endif

SRC = src/vec_graph.c src/hnsw_vtab.c src/hnsw_algo.c \
      src/graph_tvf.c src/node2vec.c src/vec_math.c \
      src/priority_queue.c src/id_validate.c

TEST_SRC = test/test_main.c test/test_vec_math.c test/test_priority_queue.c \
           test/test_hnsw_algo.c test/test_id_validate.c

# Benchmark storage: memory (default) or disk
STORAGE ?= disk

BENCH = .venv/bin/python python/benchmark_compare.py
BENCH_FLAGS = --storage $(STORAGE)

.PHONY: all debug test test-python test-all clean help \
        docs-serve docs-build docs-clean \
        benchmark-compare benchmark-analyze benchmark-clean \
        benchmark-small benchmark-medium benchmark-saturation \
        benchmark-models-prep benchmark-models

help:                                          ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

all: vec_graph$(EXT)                           ## Build the extension

vec_graph$(EXT): $(SRC)
	$(CC) $(CFLAGS_BASE) $(SHARED_FLAGS) -Isrc -o $@ $^ $(LDFLAGS)

debug: CFLAGS_BASE += -g -fsanitize=address,undefined -DDEBUG -O0
debug: LDFLAGS += -fsanitize=address,undefined
debug: vec_graph$(EXT)                         ## Build with ASan + UBSan

test: test_runner                              ## Run C unit tests
	./test_runner

test_runner: $(TEST_SRC) src/vec_math.c src/priority_queue.c src/hnsw_algo.c src/id_validate.c
	$(CC) $(CFLAGS_BASE) -Isrc -o $@ $^ $(LDFLAGS_TEST)

test-python: vec_graph$(EXT)                   ## Run Python integration tests
	.venv/bin/python -m pytest python/ -v

test-all: test test-python                     ## Run all tests

######################################################################
# BENCHMARKS
#   Override storage: make benchmark-small STORAGE=disk
######################################################################

benchmark-compare: vec_graph$(EXT)             ## Comparative benchmark vs sqlite-vector
	$(BENCH) $(BENCH_FLAGS)

benchmark-small: vec_graph$(EXT)               ## Profile: 3 dims, N≤50K, random
	$(BENCH) --source random --dim 384 --sizes 1000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 384 --sizes 5000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 384 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 384 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 1000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 5000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1536 --sizes 1000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1536 --sizes 5000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1536 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1536 --sizes 50000 $(BENCH_FLAGS)

benchmark-medium: vec_graph$(EXT)              ## Profile: 2 dims, N=100K–500K, random
	$(BENCH) --source random --dim 384 --sizes 100000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 384 --sizes 250000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 384 --sizes 500000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 100000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 250000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 500000 $(BENCH_FLAGS)

benchmark-saturation: vec_graph$(EXT)          ## Profile: 8 dims, N=50K, random
	$(BENCH) --source random --dim 32 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 64 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 128 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 256 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 512 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 768 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1024 --sizes 50000 $(BENCH_FLAGS)
	$(BENCH) --source random --dim 1536 --sizes 50000 $(BENCH_FLAGS)

benchmark-models-prep:                         ## Download models + dataset, pre-build .npy caches
	$(BENCH) --prep-models

benchmark-models: vec_graph$(EXT)  ## Profile: 3 models, N≤50K
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 1000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 5000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 50000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 1000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 5000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source model:all-mpnet-base-v2 --sizes 50000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 1000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 5000 $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 10000 $(BENCH_FLAGS)
	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 50000 $(BENCH_FLAGS)

benchmark-analyze:                             ## Analyze results → tables + plotly charts
	.venv/bin/python python/benchmark_analyze.py

benchmark-clean:                               ## Remove generated charts (keeps JSONL results)
	rm -rf benchmarks/charts/*.html benchmarks/charts/*.json

######################################################################
# DOCUMENTATION
######################################################################

docs-serve: docs-build                            ## Serve docs locally with live reload
	uv run mkdocs serve

docs-build:                                        ## Build documentation site
	uv sync --all-groups
	uv run mkdocs build --strict

docs-clean:                                        ## Clean documentation build
	rm -rf site/

######################################################################
# CLEAN
######################################################################

clean: docs-clean                                  ## Clean build artifacts
	rm -f vec_graph$(EXT) test_runner
