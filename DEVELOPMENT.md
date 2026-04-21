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

