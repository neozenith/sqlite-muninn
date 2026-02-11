######################################################################
# benchmarks/Makefile — invoke via: make -C benchmarks {target}
#   Override storage: make -C benchmarks saturation STORAGE=disk
######################################################################

PYTHON = ../.venv/bin/python
BENCH = $(PYTHON) scripts/benchmark_vss.py
BENCH_GRAPH = $(PYTHON) scripts/benchmark_graph.py
STORAGE ?= disk
BENCH_FLAGS = --storage $(STORAGE)

# Platform detection (for extension prerequisite)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    EXT = .dylib
else ifeq ($(UNAME_S),Linux)
    EXT = .so
else
    EXT = .dll
endif

EXTENSION = ../vec_graph$(EXT)

.PHONY: help compare saturation \
        models-prep models-prep-manifest models vss-analyze clean \
        graph-small graph-medium graph-large graph-scale-free graph-analyze \
        vss-manifest graph-manifest manifest analyze

help:                                              ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

$(EXTENSION):
	$(MAKE) -C .. all

######################################################################
# VECTOR BENCHMARKS
######################################################################

compare: $(EXTENSION)                              ## Comparative benchmark vs sqlite-vector
	$(BENCH) $(BENCH_FLAGS)

vss-embeddings-prep:                                       ## Download models + datasets, pre-build all .npy caches
	$(BENCH) --prep-models --prep-model MiniLM --prep-dataset ag_news
	$(BENCH) --prep-models --prep-model MPNet --prep-dataset ag_news
	$(BENCH) --prep-models --prep-model BGE-Large --prep-dataset ag_news
	$(BENCH) --prep-models --prep-model MiniLM --prep-dataset wealth_of_nations
	$(BENCH) --prep-models --prep-model MPNet --prep-dataset wealth_of_nations
	$(BENCH) --prep-models --prep-model BGE-Large --prep-dataset wealth_of_nations

vss-embeddings-prep-manifest:                              ## Show models-prep cache completeness
	$(PYTHON) scripts/benchmark_vss_analyze.py --prep-manifest

vss-embeddings: $(EXTENSION)                               ## Profile: 3 models x 2 datasets, N<=250K
	# AG News (120K passages)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH) --source model:all-mpnet-base-v2 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 1000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 5000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 10000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 50000 --dataset ag_news $(BENCH_FLAGS)
	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 100000 --dataset ag_news $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 250000 --dataset ag_news $(BENCH_FLAGS)
# 	# Wealth of Nations (~2.5K passages)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-MiniLM-L6-v2 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
# 	$(BENCH) --source model:all-mpnet-base-v2 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 1000 --dataset wealth_of_nations $(BENCH_FLAGS)
# 	$(BENCH) --source model:BAAI/bge-large-en-v1.5 --sizes 2500 --dataset wealth_of_nations $(BENCH_FLAGS)

vss-analyze:                                       ## Analyze vector results -> tables + plotly charts
	$(PYTHON) scripts/benchmark_vss_analyze.py

clean:                                             ## Remove generated charts (keeps JSONL results)
	rm -rf charts/*.html charts/*.json

######################################################################
# GRAPH BENCHMARKS
######################################################################

graph-small: $(EXTENSION)                          ## Graph: small profile (N<=1K)
# 	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100 --avg-degree 20 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 5 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 500 --avg-degree 20 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 $(BENCH_FLAGS)

graph-medium: $(EXTENSION)                         ## Graph: medium profile (N<=10K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 1000 --avg-degree 50 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 5000 --avg-degree 50 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 50 $(BENCH_FLAGS)

graph-large: $(EXTENSION)                          ## Graph: large profile (N<=100K)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 10000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 50000 --avg-degree 20 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model erdos_renyi --nodes 100000 --avg-degree 20 $(BENCH_FLAGS)

graph-scale-free: $(EXTENSION)                     ## Graph: Barabasi-Albert scale-free topology
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 3 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 5 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 1000 --avg-degree 10 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 3 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 5 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 5000 --avg-degree 10 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 3 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 5 $(BENCH_FLAGS)
# 	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 10000 --avg-degree 10 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 3 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 5 $(BENCH_FLAGS)
	-$(BENCH_GRAPH) --graph-model barabasi_albert --nodes 50000 --avg-degree 10 $(BENCH_FLAGS)

graph-analyze:                                     ## Analyze graph results -> tables + charts
	$(PYTHON) scripts/benchmark_graph_analyze.py

######################################################################
# AGGREGATED TARGETS
######################################################################

analyze: vss-analyze graph-analyze                 ## Analyze all benchmark results

vss-manifest:                                      ## Show VSS benchmark completeness
	$(PYTHON) scripts/benchmark_vss_analyze.py --manifest

graph-manifest:                                    ## Show graph benchmark completeness
	$(PYTHON) scripts/benchmark_graph_analyze.py --manifest

manifest: vss-manifest graph-manifest              ## Show overall benchmark completeness
