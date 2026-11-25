# Experimental fused complex NaN sum/count

Branch: `ianna/fused-complex-nansum`, based on PR #4549 at `82615e865`.
This is a prototype, not a replacement proposed for the main branch yet.

`awkward_reduce_nansum_count_complex` ignores an entire complex value if
either component is NaN, accumulates real and imaginary components separately
in input order, and returns the number of valid values. The CPU implementation
checks blocks of eight values in long segments (at least 128 values). Valid
blocks use an unchecked addition loop; other blocks skip invalid values. Short
segments start with ordinary summation: a NaN-free result proves all inputs
were valid. The first short row with a NaN sum is replayed with checked
additions; subsequent short rows in that invocation stay checked. The mode
resets on every call and also handles infinity cancellation. At most one
short row is replayed per call. No input-sized mask/copy or regrouping of
partial sums is introduced; long-row block reads remain cache-local. Infinities are retained. Empty and all-NaN segments have zero sum
and count. Both complex64 and complex128 use their ordinary accumulator dtype.

The canonical signature and reference are in `kernel-specification.yml`.
C++ and CUDA implementations share this contract. The CUDA implementation is a
correctness-oriented transform with one thread per segment; its callback state
is not prepared/reused yet, and long-list parallelism is not optimized.
CUDA parity tests and generated tests require a GPU and have not run locally.

The CPU matrix helper on this experimental branch uses the kernel for both nansum and unweighted
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

To compare native kernels before and after an update, first save the old
`awkward_cpp/lib/libawkward-cpu-kernels` shared library, rebuild/install the new
library, then run:

```bash
python studies/fused_complex_nansum/compare.py \
  --before-library /absolute/path/to/old/libawkward-cpu-kernels.so \
  --patterns finite strided last all --output results/fused-native-update
```

Use `.dylib` on macOS or `.dll` on Windows. The saved library must expose the
same generated signatures as the current library (the new fused symbols were
introduced at `625dcf542`). This mode keeps the Python helper identical and
loads the old native library only in baseline worker processes.
`results-native-update.md` compares the short-row update against the block
kernel at `b1aa47fbe`. `results-block-update.md` and `results-block-vs-pr.md`
retain the preceding block experiment; `results-initial.md` retains the first
fused experiment. Include `--patterns finite strided all first-row last-row
sparse-rows` to test NaNs scattered across rows as well as across elements.

Local validation: Python 3.11.14, NumPy 2.4.6, macOS arm64, locally built
awkward-cpp 59. Selected CPU/public, generated reference and generated native
suites: 770 passed, 19 existing/platform skips. Ruff and diff checks pass.
No GPU/JAX execution or GPU performance claim.

The block optimization roughly halves the first fused kernel's wide-row cost.
The subsequent per-call short-row mode reduces the measured complex64 finite
16384 x 16 nansum from 147 to 83 us against `b1aa47fbe`; complex128 falls from
135 to 87 us. These runs show substantial short-row timing variation (the
complex64 candidate process medians ranged from 81 to 114 us).

In the separate comparison with #4549 at `82615e865`, complex64 finite short
rows are still 7.5% slower (108 vs 116 us), while complex128 is 20% faster
(118 vs 94 us). Finite long rows remain about 8% slower for both dtypes in
that comparison. Do not claim these remaining regressions are eliminated.
Sparse-NaN inputs retain large gains. A first-row NaN can add a small replay
cost; a last-row NaN allows earlier rows to remain on the fast path.

These are local results, not a universal speedup; validate on additional CPUs
and GPU parity before proposing production dispatch. CUDA implementation and
package versions are unchanged by these CPU optimizations.
