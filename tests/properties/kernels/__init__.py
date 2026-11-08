# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for the compiled kernels, one module per kernel.

Each module is named after its kernel, as are the kernel's CPU and CUDA
source files, and defines a single property test, `test_matches_reference`:
on every backend in `harness.backends()`, the compiled kernel must produce
the same output as the kernel's reference implementation in
`kernel-specification.yml`, loaded with `harness.load_reference`.

Shared code is imported from `harness` (reference loading, backend
runners) and `strategies` (Hypothesis strategies). A strategy moves into
`strategies` only when a second kernel needs it; until then it stays in
its kernel's module.
"""
