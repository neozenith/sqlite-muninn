######################################################################
# benchmarks/Makefile.refactored — unified CLI-driven benchmark runner
#   invoke via: make -f benchmarks/Makefile.refactored {target}
#   or:         make -C benchmarks -f Makefile.refactored {target}
######################################################################

# Resolve project root from this Makefile's location (works regardless of CWD)
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/..)

PYTHON = uv run --directory $(PROJECT_ROOT)
BENCH = $(PYTHON) -m benchmarks.harness.cli

# ── Platform detection for extension prerequisite ──────────────────
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    EXT = .dylib
else ifeq ($(UNAME_S),Linux)
    EXT = .so
else
    EXT = .dll
endif

EXTENSION = $(PROJECT_ROOT)/build/muninn$(EXT)

.PHONY: help prep prep-status prep-vectors prep-texts prep-kg-chunks \
        prep-kg-er prep-kg-ner prep-kg-re \
        manifest manifest-missing manifest-commands \
        benchmark-vss benchmark-graph benchmark-adjacency \
        benchmark-kg benchmark-er benchmark-graphrag benchmark-node2vec benchmark-all \
        analyse analyse-docs clean clean-all extension

help:                    ## Show help
	@$(BENCH) --help

# ── Extension build (delegates to root Makefile) ──────────────────
$(EXTENSION):
	$(MAKE) -C $(PROJECT_ROOT) all

extension: $(EXTENSION)  ## Build the muninn C extension

# ── Prep ──────────────────────────────────────────────────────────
prep:                    ## Download models, texts, build caches
	$(BENCH) prep all

prep-status:                    ## Download models, texts, build caches
	$(BENCH) prep all --status

prep-vectors:            ## Download embedding models + build .npy caches
	$(BENCH) prep vectors

prep-texts:              ## Fetch Gutenberg texts
	$(BENCH) prep texts

prep-kg-chunks:          ## Create KG chunk databases
	$(BENCH) prep kg-chunks

prep-kg-er:              ## Download ER benchmark datasets
	$(BENCH) prep kg-er

prep-kg-ner:             ## Download NER benchmark datasets
	$(BENCH) prep kg-ner

prep-kg-re:              ## Download RE benchmark datasets
	$(BENCH) prep kg-re

# ── Manifest ──────────────────────────────────────────────────────
manifest:                ## Show benchmark completion status
	$(BENCH) manifest

manifest-missing:        ## Show only missing benchmarks
	$(BENCH) manifest --missing

manifest-commands:       ## Print runnable commands for missing benchmarks
	$(BENCH) manifest --missing --commands

# ── Benchmark execution (all depend on extension being built) ─────
benchmark-vss: $(EXTENSION)           ## Run all missing VSS benchmarks
	$(BENCH) manifest --missing --category vss --commands | sh

benchmark-graph: $(EXTENSION)         ## Run all missing graph benchmarks
	$(BENCH) manifest --missing --category graph --commands | sh

benchmark-adjacency: $(EXTENSION)     ## Run all missing adjacency benchmarks
	$(BENCH) manifest --missing --category adjacency --commands | sh

benchmark-kg: $(EXTENSION)            ## Run all missing KG extraction benchmarks
	$(BENCH) manifest --missing --category kg-extract --commands | sh

benchmark-er: $(EXTENSION)            ## Run all missing entity resolution benchmarks
	$(BENCH) manifest --missing --category kg-resolve --commands | sh

benchmark-graphrag: $(EXTENSION)      ## Run all missing GraphRAG benchmarks
	$(BENCH) manifest --missing --category kg-graphrag --commands | sh

benchmark-node2vec: $(EXTENSION)      ## Run all missing Node2Vec benchmarks
	$(BENCH) manifest --missing --category node2vec --commands | sh

benchmark-all: $(EXTENSION)           ## Run ALL missing benchmarks
	$(BENCH) manifest --missing --commands | sh

# ── Analysis ──────────────────────────────────────────────────────
analyse:                 ## Generate charts + tables from results
	$(BENCH) analyse

analyse-docs:            ## Generate charts + render doc templates
	$(BENCH) analyse --render-docs

# ── Cleanup ───────────────────────────────────────────────────────
clean:                   ## Remove charts (keeps results)
	rm -rf charts/*.html /charts/*.json

clean-all: clean         ## Remove everything (charts + results)
	rm -rf results/**/*.sqlite results/*.jsonl
