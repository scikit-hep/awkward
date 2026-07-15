# Phase 1 / Open Q1 — can cuda.compute compose multiple ops into one kernel?

**Short answer: Yes — but only through composition that *we* drive, not through
any native multi-primitive fusion.** cuda.compute has no deferred/lazy execution
graph that records a sequence of primitive calls and fuses them after the fact.
Each primitive invocation (`unary_transform`, `reduce_into`, `segmented_reduce`,
…) compiles and launches its own kernel immediately. What cuda.compute *does*
give you are two composition substrates that collapse many ops into the **one**
kernel of a single consuming primitive:

1. **Custom-op fusion.** Fold the whole element-wise expression into a single
   op function and hand it to one primitive. One primitive call ⇒ one kernel,
   regardless of how many arithmetic ops the op body contains.
2. **Iterator fusion.** Compose `TransformIterator` / `ZipIterator` /
   `PermutationIterator` lazily; the composed map is inlined into whatever
   primitive consumes the iterator (a transform *or* a reduction). So
   `map → map → … → reduce` becomes one kernel with **no** intermediate buffer.

So the DAG→single-kernel emitter is **ours to build** (which Phase 2 did, in
`_connect/cuda/_fusion_codegen.py`), sitting on top of cuda.compute's op- and
iterator-composition. cuda.compute compiles the composed op once and runs it as
a single kernel; it does not fuse for us across separate primitive launches.

## Which layer emits the fused kernel

The composition layer — our codegen — decides the fusion boundary; cuda.compute
is the compiler/executor for whatever single op or iterator chain we hand it:

| Region                    | What we hand cuda.compute                         | Kernels |
|---------------------------|---------------------------------------------------|:-------:|
| element-wise chain        | 1 `unary_transform` + a folded op over `ZipIterator` | 1 |
| element-wise → reduction  | 1 `segmented_reduce` fed a `TransformIterator`    | 1 |
| N separate primitive calls| N `unary_transform`s writing N intermediates      | N (+1) |

## Measured (probe on della-gpu A100, 2M float64, 100 reps)

`probe_fusion_mechanism.py`, map-only chain, time per op (ms):

| depth | A separate | B folded op | C stacked iters | A/B |
|------:|-----------:|------------:|----------------:|----:|
| 1     |      0.046 |       0.050 |           0.117 | 0.9x |
| 2     |      0.079 |       0.049 |           0.155 | 1.6x |
| 4     |      0.153 |       0.049 |           0.238 | 3.1x |
| 8     |      0.312 |       0.047 |           0.388 | 6.6x |
| 16    |      0.618 |       0.048 |           0.699 | 12.9x |

**B (one folded op) is dead flat — 0.049 ms at every depth — while A scales
linearly.** That flat line across a 16× increase in arithmetic is one kernel;
14.9× at depth 16. Definitive.

Map+reduce chain (chain then sum), time per op (ms):

| depth | A separate+reduce | B folded-op→reduce | C stacked iters→reduce | A/B |
|------:|------------------:|-------------------:|-----------------------:|----:|
| 1     |             0.157 |              0.177 |                  0.173 | 0.9x |
| 4     |             0.312 |              0.177 |                  0.301 | 1.8x |
| 16    |             0.915 |              0.176 |                  0.789 | 5.2x |

**B folded-op→reduce is flat too — 0.177 ms at every depth (5.2× at depth 16)**
— and nsys shows it fires *no separate transform kernel*: the folded map rides
inside the reduce. The probe uses `reduce_into` (whole-array reduction);
`_fusion_codegen` applies the **same** folded-op-into-reduce pattern with
`segmented_reduce` for the per-sublist case (`TransformIterator(zipped, op)` →
`segmented_reduce`). Same fusion, jagged primitive; validated end-to-end by the
`chain → sum` case in `bench_lazy_fusion.py` (cuda hit = 1, match = yes).

### nsys kernel count — exact reconciliation

`nsys profile -t cuda --stats=true` gives the kernel *count*, not just timing,
and it matches the predicted launches to the kernel:

- **9,090 transform kernels total**, exactly as predicted if the separate path
  (A) launches N transform kernels per call and the composed op (B) / iterator
  (C) launch exactly one each. All of A's separate transforms share one op
  signature and collapse into a single 7,878-instance bucket — that bucket *is*
  the separate path, and it dominates GPU time (80%).
- **`reduce_into` is a two-pass CUB reduce** — each call shows up as one
  `DeviceReduceKernel` (partial) + one `DeviceReduceSingleTileKernel` (final):
  1,515 reduce calls → 1,515 + 1,515 kernels, exactly as counted.
- **The map fused into the reduce.** In the map+reduce cases the
  `TransformIterator` fed to `reduce_into` (both the folded-op B and the
  stacked-iterator C) produced **no separate transform kernel** — the map rode
  inside the reduce kernel. Genuine map-into-reduce fusion, matching the
  `_compute.py` idiom.

### The nuance nsys clarified: launch count vs kernel quality

Both B (folded op) and C (stacked `TransformIterator`s) launch **exactly one**
kernel per call — nsys confirms C is one transform kernel per call, not N. So
stacking iterators *does* fuse to one launch. The difference is kernel
*quality*, not count: B compiles to a **bandwidth-bound** kernel where the
arithmetic hides under the 16 MB read + 16 MB write, so it stays flat as depth
grows; C's single kernel gets more expensive with nesting depth (its per-element
work isn't hidden as well). So the rule for the codegen is *how* to make the one
kernel, not *whether*: **fold the whole expression into one op** (B) for the
bandwidth-bound, depth-flat kernel — which is exactly what `_fusion_codegen`
does. Element-wise → one `unary_transform` over a folded op; transform+reduce →
one folded map op in a single `TransformIterator` into the reduce (one map
layer, not N stacked). The probe's map+reduce table now includes this folded-op
reduce (column B) alongside the stacked-iterator reduce (C).

## Evidence (three independent sources)

1. **API semantics.** Primitives are eager: each call launches. There is no
   `cuda.compute.Graph`/deferred-op API to fuse N launches post-hoc. Fusion
   only happens *inside* one primitive, via its op and its input iterator.

2. **Already shipping in-tree.** `_connect/cuda/_compute.py` fuses a map into a
   reduce with exactly this idiom: `_widen_for_reduce` / `_nonzero_for_reduce`
   wrap the input in a `TransformIterator` so "the reduction never sees (or
   allocates) a materialised copy," and `awkward_reduce_min_size` composes
   `ZipIterator + TransformIterator + reduce_into` as a single pass. This is a
   working, tested proof that iterator composition ⇒ one kernel, no intermediate.

3. **Measured (Phase 2, della-gpu A100).** Folding a depth-N element-wise chain
   into one op gives a kernel whose time is **~flat vs N** (0.19 ms at 2 ops →
   0.38 ms at 32 ops) while the per-op interpreter scales linearly (2.2 → 34 ms)
   — the signature of one kernel doing more work in registers, not N launches.
   The `chain → sum` case fuses map+reduce into one kernel and matches the
   interpreter numerically. See `bench_lazy_fusion_results.md`.

## Confirm it yourself

`studies/cccl/probe_fusion_mechanism.py` isolates the mechanism: it times a
depth-N chain run three ways (N separate primitives vs one composed op vs nested
`TransformIterator` into one consumer) across depths, for map-only and
map+reduce. If the composed-op / iterator paths stay flat while the separate
path scales with depth, composition ⇒ one kernel is confirmed. For an exact
kernel *count*:

```bash
nsys profile -t cuda --stats=true python studies/cccl/probe_fusion_mechanism.py
```

Expected: separate ≈ N transform kernels (+1 reduce); composed-op = 1;
iterator = 1.

## Consequence for the roadmap

Open Q1 is resolved: **build the emitter ourselves** on cuda.compute's
composition primitives — done in Phase 2. This is also why Phase 3 can stay on
the incumbent iterator ABI: the fused kernel consumes leaves by iterator
composition (`ZipIterator` / `TransformIterator`) with no struct parameter and
no copies. The only thing to keep watching with NVIDIA (Phase 6) is whether a
future `cuda.compute` gains a first-class deferred-graph fuser — if it does, the
codegen could target it instead of hand-composing, but nothing blocks on it.
