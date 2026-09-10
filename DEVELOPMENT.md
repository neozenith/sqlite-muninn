# Development

This is a mixed language project and at the core are `Makefile`s and `scripts/generate_build.py`.

## Overall Development

```sh
make format # Actually triggers format-c format-js and format-python
make lint # Actually triggers lint-c lint-js lint-python
make typecheck # Actually triggers typcheck-js and typecheck-python
make test # Actually triggers test-c test-js and test-python
```

```sh
make ci # Actually runs lint typecheck test test-python test-js docs-build
```

## Outer Dev Loop: `make ci-all`

Before a PR is ready for review, run the full pipeline locally. It mirrors what
GitHub Actions runs, plus the subproject CIs that the main workflow does not cover:

```sh
make ci-all # ci + ci-benchmarks-harness + ci-benchmarks-demo-builder + ci-benchmarks-sessions-demo + ci-viz
```

`make ci` alone covers the published package (`lint typecheck test test-python test-js docs-build`).
Two gates inside it are worth knowing about:

- `make test-c` fails if C line coverage (gcovr) drops below 50%. Any new `src/*.c`
  file that the test runner links must ship with a `test/test_<module>.c` suite,
  and `scripts/generate_build.py` `TEST_LINK_SOURCES` must list the file.
- `make docs-build` depends on `version-stamp`, which runs `npm audit --audit-level high`.
  A high-severity advisory in `npm/` fails the whole docs build; fix it with
  `npm --prefix npm audit fix` (never `--force` without checking `make test-js`).

## Updating llama.cpp

`vendor/llama.cpp` is a git submodule built into static libraries by CMake. To
move it to the latest upstream commit:

```sh
make llama-status  # fetch upstream and show what is new since the pinned commit
make llama-update  # update the submodule pointer and commit "chore: update llama.cpp submodule to latest"
make llama-clean   # discard the stale static libraries (they are NOT rebuilt automatically)
make all           # rebuild llama.cpp libraries and the extension
make test-c        # the wrappers in src/llama_*.c track llama.h; API drift shows up here first
```

`make llama-clean` is mandatory after an update: the Makefile only builds the
static libraries when they are missing, so a stale `vendor/llama.cpp/build/`
silently links old code against new headers. When `llama.h` removes or renames a
field (for example `use_mmap` became `load_mode = LLAMA_LOAD_MODE_MMAP`), fix
`src/llama_embed.c` and `src/llama_chat.c` in the same PR as the submodule bump.

## Releasing

`VERSION` is the single source of truth. Every other version string is stamped
from it; never edit them by hand.

1. Set the new version, following [SemVer](https://semver.org/): a new SQL
   function or TVF is a minor bump, a fix is a patch, a pre-release carries an
   `-rc1` / `-beta.1` suffix (the Publish workflow marks those as prereleases).

   ```sh
   echo "0.6.0" > VERSION
   ```

2. Stamp it everywhere and regenerate the npm lockfile:

   ```sh
   make version-stamp
   ```

   This writes `npm/package.json` (including the `@sqlite-muninn/*` platform
   sub-packages), `.claude-plugin/marketplace.json`, `.codex-plugin/plugin.json`,
   then runs `npm audit` and `npm install` so `npm/package-lock.json` matches.
   An out-of-date lockfile is the usual cause of `npm ci` failing in the
   "Code Quality" CI job.

3. Commit the stamped files with the feature PR (`chore: bump version to X.Y.Z`).
   `CHANGELOG.md` is generated from conventional commits by `make changelog`
   (git-cliff); do not hand-edit it.

4. After the PR merges, trigger the **Publish** workflow manually
   (`workflow_dispatch` on `main`). It reads `VERSION`, creates the `vX.Y.Z` tag
   and a draft GitHub release, builds every platform, and publishes to PyPI and
   npm. It fails fast if the tag already exists.

## C Developement

<details><summary><b>Expand here for full details...</b></summary>


### Inner Dev Loop

```sh
make format-c
make lint-c
make test
```

- `src/*.{h,c}`
- `test/test_*.{h,c}`
- `vendor/llama.cpp/`
- `vendor/yyjson/`

```sh
make build
```

- `build/muninn.{dylib,so,dll}`


### Distribution Packaging

```sh
make dist-extension
```

- `dist/muninn.{dylib,so,dll}`

```sh
make amalgamation
```

- `dist/muninn.h`
- `dist/muninn.c`

```sh
make generate-windows
```

- `build/generated/build_windows.bat`

Needed to build the window library on Github Actions

</details>

## Python Development

Thin wrapper around C library for python use.

<details><summary><b>Expand here for full details...</b></summary>

- `sqlite-muninn/`
- `pytests/`

```sh
make format-python
make lint-python
make typecheck-python
make test-python
```

</details>

## JS Development

Thin wrapper around C library for Node.JS use as well as the WASM build.

<details><summary><b>Expand here for full details...</b></summary>

- `npm/src/*.js`
- `npm/src/*.test.js`

```sh
make format-js
make lint-js
make typecheck-js
make test-js
```
</details>

