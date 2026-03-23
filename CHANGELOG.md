# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Bug Fixes
- Address CI memory leak not unloading models. Address CI build issues needing to propagate from main build to windows build script.- Demo db builder fully functional- Remove FTS5Adapter from kg benchmarks- Removed FTS5Adapater from KG benchmarks- Llm_extract example compares a few models performing the NER and RE benchmarked against GLiNER2 models and the honest speed comparrisons- Refactored viz demo- Notebook kernel CWD is examples/{name}/, not project root- Add --cache-bust flag to Colab E2E test script

### CI
- Set concurrency groups to cancel in flight builds when a newer commit is available and set timeouts to cap jobs that are hanging in CMake.- Refactor sources and build targets into one centralised spot- Refactor wasm build pipeline and rebuild the kg-demo database for wasm/ and viz/

### Documentation
- Update docs with benchmark results for VSS for 50k and 100K embeddings- Update planning documents- Huge refactor of benchmarking pipeline to consolidate duplicated code across benchmarking and analysis tasks- Update the text embedding example README- Start the embedding benchmark docs pages- Ran all the ag-news benchmarks for embed category- Update graph benchmakrs- Update feature list in README and mkdocs index- Update feature list again to include dbt-syntax graph selection- Refactor mermaid diagrams to hires pngs. clean out old specs- Finalised the kg-demo.db builder script- Update demo builder plan- Update plan for wasm+viz merger and demo_builder- Add documentation about the logo tooling to remove the background

### Features
- Full refactor of benchmarking prep tasks- Adding llama.cpp integration- Refactored the vss benchmark pipeline to use GGUF models for embeddings to be consistent with the impending embed category of benchmarks- Updated demo_builder and session_demo for narrowest context window for kg pipelines- Benchmarks.sessions_demo to build incremental knowledge graph from claude code sessions files. Lots of speed tuning splitting tasks into fine grained steps to find bottlenecks, making models work offline without needing constant internet checks or unnecessary redownloads of models.- Refactor sessions_demo and demo_builder to add GLiNER2 backend, and incremental UMAP for demo pipelines- Added llama.cpp chat and enhanced the NER and RE tasks- Improve demo builder build subcommands.- Refactor llama_common out of llama_embed and llama_chat. Added muninn_tokenize_text, also improved the muninn_summarize- Add Colab notebook generation and README badge enforcement- Rename examples to {name}.py, add nbmake test targets- Add 3-environment path resolution for Colab + E2E test script

### Other
- Update dev tooling script for logo image processing to spit out full sequential step explanation.- Huge refactor of demo builder- Add Claude Code GitHub Workflow (#1)

* "Claude PR Assistant workflow"

* "Claude Code Review workflow"- Updated plan docs- Use qwen3.5 in example

### Refactor
- Refactor benchmarks cli usage docs, add updated plannign docs for next phases.

## [0.2.0] - 2026-02-18

### Bug Fixes
- Update amalgamation script with new files- Address windows amalgamation script and name collision in amalgamation source code

### Documentation
- Updated refinement of upcoming plan documents

### Features
- Feat (graph): Graph adjacency virtual table with lazy incremental rebuild- Implemented dbt graph selector syntax tvf

## [0.1.0] - 2026-02-18

### Bug Fixes
- Improved visualisation hover text on embedding vis- Sanitise fts query strings in demo visualisations

### Documentation
- Hardcode absolute URL instead of relative URL to example to bypass mkdocs link resolver

### Other
- V0.1.0 release

## [0.1.0-alpha.10] - 2026-02-17

### Bug Fixes
- Refactor npm deployment to generate platform specific package.json and refined the wasm/ and viz/ demo servers- Address ci code formatting issues

## [0.1.0-alpha.9] - 2026-02-17

### Bug Fixes
- Update the publish.yml to update the package-lock.json automatically in publishing but attempt to try to pre-resolve during make version-stamp target

## [0.1.0-alpha.8] - 2026-02-17

### Bug Fixes
- Address more npm publishing bugs

## [0.1.0-alpha.7] - 2026-02-17

### Bug Fixes
- Address build and publish sequence and tsup prePublishOnly hook not having devDependencies available

## [0.1.0-alpha.6] - 2026-02-16

### Bug Fixes
- Iterating on github action publishing to npm with trusted publishing

## [0.1.0-alpha.5] - 2026-02-16

### Bug Fixes
- Deploying to sqlite-muninn npm org instead- Need to specify --tag when publishing to npm

## [0.1.0-alpha.2] - 2026-02-16

### Bug Fixes
- Deploy multi-platform binaries to npm

### Documentation
- Fixed the linked logo image in the readme for pypi and npm

## [0.1.0-alpha.1] - 2026-02-16

### Bug Fixes
- Fix CI: use pysqlite3-binary for extension loading, install uv

The actions/setup-python@v5 Python 3.13 builds lack
PY_SQLITE_ENABLE_LOAD_EXTENSION, so enable_load_extension() is
unavailable. Use pysqlite3-binary as a drop-in replacement in CI.
Also install uv via astral-sh/setup-uv@v4 for package install tests,
and fix the persistence test's extension path to use build/muninn.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>- Fix CI: pysqlite3 fallback for ARM64/macOS, add build/ to npm binary search

pysqlite3-binary only publishes wheels for Linux x86_64. Fall back to
pysqlite3 (source compile) on ARM64 and macOS where no binary wheel
exists.

Also add build/ directory to getLoadablePath() search order since
make all outputs to build/muninn.{so,dylib,dll}, not the repo root.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation
- Doc (benchmarks): Rebuilding the benchmarked dataset- Doc (benchmark): Rebuilding the benchmark dataset

### Features
- Add WASM demo, overhaul viz frontend, refine CD plan- Add publish.yml workflow for release automation- Build platform wheels natively in CI with uv

### Other
- Initial commit- Initial benchmarking results- Updated benchmark metrics results- More benchmark docs updates- Updated docs benchmarks- Update docs dataset URL reference- Checkpoint planning documents- Project rename to sqlite-muninn- Add project logo- Add graph community and centrality- Updated planning documents- Add ci and agent skills as well as python and nodejs wrappers- Some house keeping- Huge CI refactor- Iterate on fixing CI- Started works on visualisation tool and planning out KG benchmarks

### Refactor
- Refactored some more of the manifests architecture

<!-- generated by git-cliff -->
