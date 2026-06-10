# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout: two packages

This repo ships **two PyPI packages** from one repository:

- **`awkward`** — pure Python, in `src/awkward`. Released frequently.
- **`awkward-cpp`** — compiled C++ kernels (pybind11 + scikit-build-core), in `awkward-cpp/`. Versioned independently, released rarely (compilation takes hours in CI). `pyproject.toml` pins the exact `awkward-cpp` version that `awkward` requires.

Because parts of `awkward-cpp` are generated, a fresh clone will not build until you run the `prepare` step.

## Development setup

```bash
git clone --recursive https://github.com/scikit-hep/awkward.git  # rapidjson submodule
nox -s prepare                      # generate headers, kernel signatures, kernel tests, docs data
python -m pip install -v ./awkward-cpp
python -m pip install -e .
```

- `noxfile.py` is a uv script (`./noxfile.py -s prepare` also works without nox installed).
- `nox -s prepare -- --headers --signatures --tests --docs` runs selected generation targets only; `nox -R -s ...` reuses the session venv.
- Python-only changes need no rebuild (editable install). Rebuilding `awkward-cpp` is only needed when C++ changes; for fast iteration install build deps and use `pip install --no-build-isolation --check-build-dependencies ./awkward-cpp`.

## Testing

```bash
python -m pytest -n auto tests                      # main test suite
python -m pytest tests/test_XXXX-description.py     # single test file
python -m pytest -n auto awkward-cpp/tests-spec         # generated kernel-spec tests (Python reference impls)
python -m pytest -n auto awkward-cpp/tests-cpu-kernels  # generated tests against compiled kernels
```

CUDA suites (`tests-cuda`, `tests-cuda-kernels`) require an Nvidia GPU + CuPy. The generated kernel test dirs only exist after `nox -s prepare` (the `--tests` target).

Test files are named `tests/test_XXXX-description.py` where `XXXX` is the GitHub issue or PR number (enforced by `dev/validate-test-names.py`). New code needs a new test file following this convention.

Warnings are errors in pytest (`filterwarnings = ["error", ...]`).

## Linting

```bash
pre-commit run -a    # or: nox -s lint    (user preference: prek -a --quiet)
```

There is also a custom flake8 plugin (`dev/flake8_awkward.py`) and `nox -s pylint`.

## Architecture

### Layered Python design (`src/awkward`)

- `highlevel.py` — user-facing `ak.Array`, `ak.Record`, `ak.ArrayBuilder`. These wrap a low-level layout.
- `operations/ak_*.py` — one file per public `ak.*` function. Each goes through `_dispatch.py` (`@high_level_function`), which enables third-party overload dispatch; the function body is split into a `dispatch` generator yielding array arguments, then the `_impl` function.
- `contents/` — the layout node types (`NumpyArray`, `ListOffsetArray`, `RecordArray`, `IndexedOptionArray`, `UnionArray`, etc.), forming a tree that represents nested/ragged data columnar-wise. `index.py` holds the integer index buffers. `_do.py` contains cross-cutting operations on layouts.
- `forms/` — metadata-only mirrors of each content type (the "form" = type + buffer structure without data), used for serialization and typetracer.
- `_meta/` — shared base logic between contents and forms.
- `_backends/` and `_nplikes/` — abstraction over array libraries: NumPy, CuPy, JAX, and **typetracer** (shape-only arrays with possibly-unknown lengths, used by dask-awkward to compute without data). Code in `contents/` must go through the nplike API, not call NumPy directly.
- `_kernels.py` + `awkward_cpp._kernel_signatures` — how layouts invoke compiled kernels via ctypes-style signatures, per backend.
- `_connect/` — integrations: numba, jax, pyarrow (Arrow/Parquet), numexpr, RDataFrame, cling/cppyy, cuda.
- `behaviors/` — built-in behaviors (e.g. strings as character lists) layered on the `ak.behavior` registry, which maps record names/parameters to Python mixin classes.
- `_broadcasting.py`, `_slicing.py`, `_reducers.py` — the core algorithms behind ufuncs, `__getitem__`, and reductions.

### Kernels are spec-driven

`kernel-specification.yml` (repo root) is the source of truth for every low-level kernel: signatures plus a Python reference implementation. From it:

- `dev/generate-kernel-signatures.py` → `awkward-cpp/include/awkward/kernels.h`, `awkward_cpp/_kernel_signatures.py`, and the CUDA signature table in `src/awkward/_connect/cuda/`.
- `dev/generate-tests.py` + `kernel-test-data.json` → the `tests-spec*` and `tests-cpu-kernels*` suites.

The hand-written C++ implementations live in `awkward-cpp/src/cpu-kernels/` (one file per kernel) and must match the spec; `nox -s diagnostics` checks kernel definitions. Adding/changing a kernel means touching the YAML spec, the C++ implementation, and (sometimes) test data, then re-running `nox -s prepare`.

### header-only C++

`header-only/` contains standalone C++ headers (`LayoutBuilder`, `GrowableBuffer`) used by downstream C++ projects; `dev/copy-cpp-headers.py` (part of `prepare`) copies them into `awkward-cpp` and `src/awkward/_connect/header-only`. Never edit the copies.

## Conventions and gotchas

- PR titles follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...) — PRs are squash-merged, so the title becomes the commit message. Branches are named `<github-userid>/description`.
- `main` must always be in a working state and stays close to the latest release. Do not commit directly to `main` (PRs only). `main-v1` is a separate diverged branch for Awkward 1.x fixes.
- Changes to core dependencies (NumPy, Numba, Pandas, PyArrow) or Python version support must be discussed with maintainers first.
- If a PR requires C++ changes, the `awkward-cpp` version must be bumped in both `awkward-cpp/pyproject.toml` and the pin in `pyproject.toml`, and an `awkward-cpp` release made before the `awkward` release.
- AI-assisted contributions must be disclosed in the PR description (see CONTRIBUTING.md).

## Docs

```bash
nox -s prepare -- --docs --headers
nox -s docs -- --serve    # requires Doxygen; serves at localhost:8000
```
