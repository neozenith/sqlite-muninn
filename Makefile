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

.PHONY: all debug test test-python test-all clean help \
        docs-serve docs-build docs-clean

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
	.venv/bin/python -m pytest pytests/ -v

test-all: test test-python                     ## Run all tests

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
