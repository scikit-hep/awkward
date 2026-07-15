# Lazy IR kernel fusion — benchmark results

Reproduce with:

```bash
python studies/cccl/bench_lazy_fusion.py --out bench.json
```

## What is measured

Every op in the operator IR is, in the eager/interpreter model, its own
dispatch + its own materialized intermediate — one kernel launch per op on a
GPU. The fusion pass collapses a maximal element-wise region (see
`_connect/lazy/_fusion.py`) into a single `FusedNode`, which a backend
realizes as **one** kernel:

- **CUDA** (`_connect/cuda/_fusion_codegen.py`): one `cuda.compute` transform
  over a `ZipIterator`, or a `TransformIterator` fed into `segmented_reduce` —
  intermediates stay in registers/L1.
- **CPU** (`_connect/cpu/_fusion_codegen.py`): one pass over the shared flat
  `NumpyArray` content, so the per-op `ak.Array` dispatch/broadcast cost is
  paid **once** instead of N times, and only the final result is rewrapped.

Two numbers per case: the **launch proxy** (element-wise ops → fused kernels,
backend-independent) and **CPU wall-clock** (`fuse=False` interpreter vs
`fuse=True`). The GPU on-chip-reuse win is not measurable on this machine (no
GPU); only the launch-count reduction is projected.

## Results (this machine, NumPy backend)

### 200,000 lists / 600,016 items — dispatch-bound

| case            | ew ops | fused kernels | interp ms | fused ms | speedup |
|-----------------|-------:|--------------:|----------:|---------:|--------:|
| chain d=1       |      2 |             1 |      1.14 |     0.21 |  5.46x  |
| chain d=2       |      4 |             1 |      2.21 |     0.36 |  6.11x  |
| chain d=4       |      8 |             1 |      4.57 |     0.65 |  7.04x  |
| chain d=8       |     16 |             1 |      9.07 |     1.24 |  7.30x  |
| chain d=16      |     32 |             1 |     18.51 |     2.20 |  8.40x  |
| filter pipeline |      3 |             2 |      8.37 |     7.14 |  1.17x  |

### 2,000,000 lists / 6,001,426 items — bandwidth-bound

| case            | ew ops | fused kernels | interp ms | fused ms | speedup |
|-----------------|-------:|--------------:|----------:|---------:|--------:|
| chain d=1       |      2 |             1 |      4.30 |     2.60 |  1.65x  |
| chain d=2       |      4 |             1 |      8.67 |     4.15 |  2.09x  |
| chain d=4       |      8 |             1 |     18.11 |     7.19 |  2.52x  |
| chain d=8       |     16 |             1 |     36.31 |    13.36 |  2.72x  |
| chain d=16      |     32 |             1 |     73.21 |    25.43 |  2.88x  |
| filter pipeline |      3 |             2 |     59.76 |    58.05 |  1.03x  |

## Reading the numbers

- **Fusion collapses any-depth element-wise chain to one kernel.** A 16-op
  chain (32 element-wise nodes) becomes a single fused region — 31 fewer
  dispatches/intermediates. That reduction is exactly what removes kernel
  launches on a GPU.
- **CPU speedup grows with chain depth** (5.5x → 8.4x at 200k) because each
  fused op removes one round-trip through `ak`'s dispatch/broadcast layer;
  deeper chains amortize more of it.
- **Smaller/dispatch-bound arrays win most** (200k: up to 8.4x) — fixed
  per-op overhead dominates there. At 2M items the arithmetic is more
  memory-bandwidth-bound, so the relative win narrows to ~1.6–2.9x (still
  real: fewer temporaries → less memory traffic). This matches the follow-up
  plan's guidance that fusion pays off most on launch/dispatch-bound work.
- **The filter pipeline barely moves** (1.03–1.17x): the boolean-mask filter
  itself is a structural op that fusion (correctly) does not fuse, so it
  dominates. Honest boundary of the technique.
- **Fused reductions also run where the eager path can't.** `sum`/`mean` are
  computed on the flat buffer with NumPy, so they work even on a machine whose
  eager `ak.sum` kernel is unavailable.

## GPU arm — measured on della-gpu (A100), cuda.compute

On a machine with `cupy` + `cuda.compute` the benchmark runs a CUDA arm that
exercises the real single-kernel codegen (`_connect/cuda/_fusion_codegen.py`):

```bash
python studies/cccl/bench_lazy_fusion.py --gpu on --out bench.json
```

### Results (della-gpu). `cuda hit` = regions that became a real kernel; `match` = equals interpreter

**200,000 lists / 600,016 items**

| case            | ew ops | kernels | cuda hit | match | interp ms | fused ms | speedup |
|-----------------|-------:|--------:|---------:|:-----:|----------:|---------:|--------:|
| chain d=1       |      2 |       1 |        1 |  yes  |     2.233 |    0.191 |  11.72x |
| chain d=2       |      4 |       1 |        1 |  yes  |     4.292 |    0.204 |  21.06x |
| chain d=4       |      8 |       1 |        1 |  yes  |     8.609 |    0.229 |  37.63x |
| chain d=8       |     16 |       1 |        1 |  yes  |    17.160 |    0.280 |  61.39x |
| chain d=16      |     32 |       1 |        1 |  yes  |    34.239 |    0.378 |  90.68x |
| chain d=4 →sum  |      8 |       1 |        1 |  yes  |     9.579 |    0.932 |  10.27x |
| filter pipeline |      3 |       2 |        2 |  yes  |     6.471 |    3.580 |   1.81x |

**2,000,000 lists / 6,001,426 items**

| case            | ew ops | kernels | cuda hit | match | interp ms | fused ms | speedup |
|-----------------|-------:|--------:|---------:|:-----:|----------:|---------:|--------:|
| chain d=1       |      2 |       1 |        1 |  yes  |     2.168 |    0.189 |  11.48x |
| chain d=2       |      4 |       1 |        1 |  yes  |     4.273 |    0.203 |  21.02x |
| chain d=4       |      8 |       1 |        1 |  yes  |     8.544 |    0.229 |  37.37x |
| chain d=8       |     16 |       1 |        1 |  yes  |    17.005 |    0.280 |  60.83x |
| chain d=16      |     32 |       1 |        1 |  yes  |    34.141 |    0.379 |  89.98x |
| chain d=4 →sum  |      8 |       1 |        1 |  yes  |     9.592 |    0.689 |  13.93x |
| filter pipeline |      3 |       2 |        2 |  yes  |     8.360 |    4.858 |   1.72x |

### What the GPU numbers prove

- **The launch/on-chip story is real, not projected.** The interpreter time
  scales linearly with op count (2 ops 2.2 ms → 32 ops 34 ms — one launch +
  one global-memory round-trip per op). The **fused time is nearly flat**:
  0.19 ms at 2 ops, 0.38 ms at 32 ops. A 16× longer chain costs ~2× more,
  because it is one kernel doing more arithmetic in registers — intermediates
  never touch global memory. That flat line *is* kernel fusion.
- **Speedup is size-independent on GPU** (~11–91x at both 200k and 2M),
  because the win is launch elimination + register reuse, not the per-op host
  dispatch that dominates the CPU path (and washes out when the CPU case turns
  bandwidth-bound at 2M). Deeper chains win more: **up to ~91x** at depth 16.
- **Transform+reduce fuses into one kernel and is correct.** `chain d=4 →sum`
  runs the element-wise map and the `segmented_reduce` as a single kernel
  (`cuda hit = 1`, `match = yes`), 10–14x over the interpreter.
- **Every case matched the interpreter** (`match = yes`) with `cuda hit ≥ 1` —
  the device codegen engaged on every fusible region and never silently fell
  back, so this run doubles as an end-to-end correctness test of the CUDA
  lowering.
- **Even the partially-fusible pipeline wins on GPU** (1.7–1.8x vs ~1.03x on
  CPU): its two fused regions each remove device launches even though the
  boolean-mask filter between them is not fused.

For each case it reports, alongside device timings (measured with CUDA events,
warmed up to exclude NVRTC/JIT):

- `cuda hit` — how many fused regions actually became a `cuda.compute` kernel
  (via `executor.fused_hits["cuda"]`).  `0` means the region fell back to eager
  evaluation — printed as a loud warning so a silent fallback can't be mistaken
  for "fusion didn't help".
- `match` — whether the fused result equals the interpreter result
  (`allclose`, tolerant to float reduction-order drift).  So the arm is also a
  correctness test of the device codegen.

Cases: the same element-wise chains (`unary_transform` over a `ZipIterator`),
a `chain d=4 -> sum` case that fuses the map into a `segmented_reduce` (one
kernel, no intermediate buffer), and the filter pipeline.  Each case is
isolated — a failure is reported in its row rather than aborting the sweep, so
the first real device run surfaces any `cuda.compute` API mismatch cleanly.

## GPU projection (CPU-only fallback)

Across the cases above, fusion removes **116 kernel launches**. At a
conservative ~6 µs/launch pure overhead that is ~0.70 ms — and that is *before*
the larger GPU win the interpreter can't avoid: keeping every intermediate in
registers/L1 instead of writing it back to global memory between ops. The
follow-up plan's Nsight trace shows the eager dimuon-mass chain firing a long
sequence of ~0.3–1.2 ms launches with materialized intermediates; that per-op
launch+materialize chain is what one fused kernel replaces.
