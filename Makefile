CC ?= cc
CFLAGS_BASE = -O2 -Wall -Wextra -Wpedantic -std=c11 -fPIC
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

.PHONY: all debug test test-python test-all clean help \
        amalgamation install uninstall skill version \
        docs-serve docs-build docs-clean

help:                                          ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

version:                                       ## Print version
	@echo $(VERSION)

######################################################################
# BUILD
######################################################################

all: muninn$(EXT)                              ## Build the extension

muninn$(EXT): $(SRC)
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) $(SHARED_FLAGS) -Isrc -o $@ $^ $(LDFLAGS)

debug: CFLAGS_BASE += -g -fsanitize=address,undefined -DDEBUG -O0
debug: LDFLAGS += -fsanitize=address,undefined
debug: muninn$(EXT)                            ## Build with ASan + UBSan

######################################################################
# TEST
######################################################################

test: test_runner                              ## Run C unit tests
	./test_runner

test_runner: $(TEST_SRC) src/vec_math.c src/priority_queue.c src/hnsw_algo.c src/id_validate.c src/graph_load.c
	$(CC) $(CFLAGS_BASE) $(CFLAGS_EXTRA) -Isrc -o $@ $^ $(LDFLAGS_TEST)

test-python: muninn$(EXT)                      ## Run Python integration tests
	.venv/bin/python -m pytest pytests/ -v

test-all: test test-python                     ## Run all tests

######################################################################
# PACKAGING
######################################################################

amalgamation: dist/muninn.c dist/muninn.h      ## Create single-file amalgamation

dist/muninn.c dist/muninn.h: $(SRC) $(HEADERS)
	bash scripts/amalgamate.sh

skill:                                         ## Stamp version into skill files
	@VERSION=$$(cat VERSION); \
	mkdir -p dist/skills/muninn/references; \
	for f in skills/muninn/SKILL.md skills/muninn/references/*.md; do \
	    [ -f "$$f" ] && sed "s/{{VERSION}}/$$VERSION/g" "$$f" > "dist/$$f"; \
	done

######################################################################
# INSTALL
######################################################################

PREFIX ?= /usr/local

install: muninn$(EXT)                          ## Install extension and header
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 muninn$(EXT) $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/include
	install -m 644 src/muninn.h $(DESTDIR)$(PREFIX)/include/

uninstall:                                     ## Remove installed files
	rm -f $(DESTDIR)$(PREFIX)/lib/muninn$(EXT)
	rm -f $(DESTDIR)$(PREFIX)/include/muninn.h

######################################################################
# DOCUMENTATION
######################################################################

docs-serve: docs-build                         ## Serve docs locally with live reload
	uv run mkdocs serve

docs-build:                                    ## Build documentation site
	uv sync --all-groups
	uv run mkdocs build --strict

docs-clean:                                    ## Clean documentation build
	rm -rf site/

######################################################################
# CLEAN
######################################################################

clean: docs-clean                              ## Clean build artifacts
	rm -f muninn$(EXT) test_runner
	rm -rf dist/
