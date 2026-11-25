# Experimental fused complex NaN sum/count

Branch: `ianna/fused-complex-nansum`, based on PR #4549 at `82615e865`.
This is a prototype, not a replacement proposed for the main branch yet.

`awkward_reduce_nansum_count_complex` traverses each offset segment once,
ignores an entire complex value if either component is NaN, accumulates real
and imaginary components separately in input order, and returns the number of
valid values. Infinities are retained. Empty and all-NaN segments have zero sum
and count. Both complex64 and complex128 use their ordinary accumulator dtype.

The canonical signature and reference are in `kernel-specification.yml`.
C++ and CUDA implementations share this contract. The CUDA implementation is a
correctness-oriented transform with one thread per segment; its callback state
is not prepared/reused yet, and long-list parallelism is not optimized.
CUDA parity tests and generated tests require a GPU and have not run locally.

The CPU matrix helper in #4549 uses the kernel for both nansum and unweighted
nanmean. It avoids the input-sized mask, zero-filled complex copy, and second
reduction. Counts preserve mask_identity and mean behavior. General ragged,
virtual, parameterized and other-backend paths are unchanged.

## Build and test

Use an isolated environment. A locally rebuilt awkward-cpp is required because
the new symbol is absent from the published library, even though package
versions deliberately remain unchanged in this experimental branch.
Initialize the rapidjson submodule in a fresh clone first.

```bash
git submodule update --init --recursive
python -m pip install ./awkward-cpp
python -m pip install -e .
python -m pytest -q tests/test_4549_complex_nan_masking.py tests/test_4549_fused_complex_nansum.py
python -m pytest -q awkward-cpp/tests-spec/*nansum_count* awkward-cpp/tests-cpu-kernels/*nansum_count*
# GPU node only:
python -m pytest -q tests-cuda/test_4549_fused_complex_nansum.py tests-cuda/test_4549_complex_nan_masking.py tests-cuda-kernels/*nansum_count*
```

The build runs signature and test generation. No generated copies are edited.
A production merge would need release coordination for the added C++ symbols;
no awkward-cpp version or Python dependency pin is changed here.

## Measurement

```bash
python studies/fused_complex_nansum/compare.py --output results/fused-complex-nansum
```

Both variants run in fresh processes using the same rebuilt native library.
The baseline restores only the CPU matrix helper from #4549 at `82615e865`;
all other source/dependencies remain identical. Three alternating pairs,
seven batches of five calls and three warmups; parity checks outside timing.
This isolates the helper change, not a whole-checkout comparison against main.
See `results-local.md`. Raw process samples are written under `--output`.

Local validation: Python 3.11.14, NumPy 2.4.6, macOS arm64, locally built
awkward-cpp 59. Selected CPU/public, generated reference and generated native
suites: 680 passed, 19 existing/platform skips. Ruff and diff checks pass.
No GPU/JAX execution or GPU performance claim.

The prototype is faster on tiny matrices and NaN-containing matrices, but
finite large matrices regress (up to about 2x in this sample). Do not promote it
as a universal optimization. Investigate the added per-element validity/count
cost while preserving summation order before proposing production dispatch.
