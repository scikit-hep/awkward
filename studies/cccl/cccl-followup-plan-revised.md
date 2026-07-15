# CUDA CCCL + Awkward Array — follow-up plan (revised)

Status: working document · Revised 2026-07 to reflect the actual repo state,
then again to absorb a strategic shift: **dask-awkward is a dead end**, so this
lazy/fusion layer is a candidate *replacement* for its role, not just a GPU
speedup — which promotes `ak.*` dispatch compatibility and typetracer-driven
lazy loading to gating requirements (see "Strategic context" and Objectives
5–6). Supersedes the original "follow-up overview," which read as greenfield;
large parts are already built or already answered in-tree. Read together with
[`cuda-compute-migration-plan.md`](./cuda-compute-migration-plan.md), which is
the ground truth for kernel-level migration status.

Related NVIDIA issues (unchanged):
NVIDIA/cccl#5496 · NVIDIA/numba-cuda#37 · NVIDIA/cccl#5556 ·
cupy/cupy#8597 · rapidsai/cudf#11983

---

## What already exists (baseline, not "to do")

Before stating goals, the plan must anchor on what is in `main` today.
Re-auditing against the code:

**Kernel migration (raw CUDA → CCCL).** Per `cuda-compute-migration-plan.md`:
108 of 143 kernel families are already compute-dispatched; ~11 remain
CuPy-dispatched (3 batch-2 + 8 Phase-1 reverts); 74 dead `.cu` files await
deletion. This is *most of Task 1 in the original plan*, already done and
benchmarked on della. The migration recipe, phase tracking, and revert
criteria live in that doc — this plan does not restate them.

**Zero-copy device access (a working ABI already ships).**
`src/awkward/_connect/cuda/helpers.py::awkward_to_cccl_iterator` builds
`ZipIterator` / `PermutationIterator` directly from `ak._do.to_buffers`,
recursing through Record / Indexed / ListOffset forms. Awkward data already
reaches cuda.compute primitives with no copies — via *iterator composition*,
not a passed struct. This is a different design from the proposed
`awkward_view_t`, and it is the incumbent.

**Lazy execution machinery (two IR layers, both backend-neutral).**
- `_connect/lazy/core.py` — low-level `IRNode` that *lowers* to CCCL helpers
  (`segmented_select`, `select_segments`, `transform_segments`). Used by
  `_connect/cuda/ir_nodes.py`. Demand-driven, memoized, stack-safe
  (iterative topological execution). Explicitly designed for a future
  `_connect/cpu/ir_nodes.py` sibling.
- `_connect/lazy/_ir.py` + `_connect/lazy/_executor.py::IRExecutor` —
  high-level operator IR (`BinaryOpNode`, `ReduceNode`, `FilterNode`, …) built
  by `LazyAwkwardArray`. **The executor is an interpreter**: each node
  dispatches to an eager Awkward op and materializes it (LRU-bounded memo,
  256 MiB default). Its docstring states the CCCL fast paths are *not* used
  here.

The net: lazy **graph construction** and **demand-driven, memoized
evaluation** exist. **Kernel fusion now also exists** (Phase 2, done — see
below): `_connect/lazy/_fusion.py` rewrites the operator-graph, collapsing
element-wise regions (and transform+reduce) into single `FusedNode` kernels;
`_connect/cuda/_fusion_codegen.py` and `_connect/cpu/_fusion_codegen.py` lower
them to one `cuda.compute` / one flat-NumPy pass. `IRExecutor.optimize()` runs
the pass; `compute(fuse=False)` is the no-fuse debug path.

---

## Data layout and the device-view ABI (AOS vs SOA)

This section settles what is actually passed to Awkward's GPU kernels, because
the ABI discussion with NVIDIA (`awkward_view_t`) hinges on it and the original
plan left it implicit.

**Awkward is struct-of-arrays (SOA), end to end.** There is no
array-of-structs representation anywhere in the data model: a `RecordArray`
stores each field as its own contiguous buffer (never `[{x,y,z}, …]`); a
`ListOffsetArray` stores `offsets` plus a flattened `content` buffer. All three
GPU paths pass *separate typed buffers*, never an interleaved struct array:

- **Raw CUDA kernels** (CuPy `RawModule`, the ~11 remaining + being retired):
  each buffer is an independent, individually-templated pointer argument, e.g.
  `awkward_ListArray_broadcast_tooffsets_a(T* tocarry, const C* fromoffsets,
  int64_t offsetslength, const U* fromstarts, const V* fromstops, …)`. Pure
  SOA; each buffer's dtype is specialized independently.
- **cuda.compute path** (`helpers.py::awkward_to_cccl_iterator`): walks the
  form and builds SOA iterators — `NumpyArray → flat CuPy buffer`,
  `RecordArray → ZipIterator(*field_iterators)`,
  `IndexedArray → PermutationIterator(content, index)`,
  `ListOffsetArray → content iterator + separate offsets buffer`.
- **Numba JIT path** (`arrayview_cuda.py`): passes `(pos, start, stop,
  arrayptrs, pylookup)`, where `arrayptrs` is a *device array of pointers* to
  the individual buffers and `pos/start/stop` are tree-navigation metadata. The
  kernel reconstructs `array.field[i]` by indexing `arrayptrs`; the payload
  stays columnar.

**Implication for cuda.compute AOS vs SOA.** `ZipIterator` (SOA) is the
natural, zero-copy fit and is what the working code already uses — records map
1:1 to `ZipIterator` precisely because their fields are already separate
buffers. `gpu_struct` (AOS) would require *interleaving* those separate buffers
into one array-of-structs buffer, a copy that defeats zero-copy; it only makes
sense for a small tuple constructed fresh, not for consuming existing Awkward
records. `ZipIterator` also gives the ergonomic win of AOS without the memory
cost: kernel authors read `p.x, p.y` (struct-*valued* iteration) while storage
stays columnar.

**Two clarifications for the NVIDIA/`awkward_view_t` discussion:**

1. **`awkward_view_t` would be *new*, not a repackaging of the current ABI.**
   Nothing today passes a single packed per-form struct like
   `struct awkward_ListArray_64_float32 { int64_t* starts; int64_t* stops;
   float* content; int64_t length; };`. Raw kernels take *N separate pointer
   args*; the Numba path passes a *flat pointer table* (`arrayptrs`) + pos tree.
   `awkward_view_t` as a single typed struct-of-pointers is a formalization
   *layered on top* — the plan should say "define," not "reuse." (Note also:
   list layouts carry `starts`/`stops` or `offsets`; `parents` is a
   reducer-side concept, not a list buffer — don't put it in the list struct.)
   Whatever `awkward_view_t` holds, it is a **struct of pointers to SOA
   columns**, not an AOS repacking of the data.

2. **There are two device-view substrates; name both so they don't conflate:**
   - `arrayview_cuda.py` → the **Numba** device view (pointer table + pos
     tree), consumed by `numba.cuda.jit` kernels.
   - `awkward_to_cccl_iterator` → the **cuda.compute** view (composed
     iterators), consumed by CUB/Thrust primitives.

   `awkward_view_t` is the proposal to unify these into one C struct both can
   consume. That unification only pays off once fusion emits *custom* kernels
   (hence ABI is sequenced after the fusion decision — see Phase 3).

**Action — expose `awkward_to_cccl_iterator` as a supported (internal) entry
point.** The PyHEP-talk prototype (shwina/awkward-cccl) grabs buffers out of the
`ak.Array` and constructs iterators by hand, and explicitly wishes that were
"tucked away within Awkward." It already is: `awkward_to_cccl_iterator` does
exactly this (zero-copy, recursive, returns the iterator + an offsets/length/
count metadata dict). It is wrapped as `awkward._connect.cuda.to_cccl_iterator`.
Keep it in `_connect` for now — while the lazy/fusion work is a PoC it should
*not* be blessed as a public `ak.*` name (see Phase 3); external CCCL code can
still call the `_connect` helper instead of hand-rolling buffer extraction.

**Why fusion is the prize (concrete evidence).** The Nsight trace of the eager
dimuon-mass computation in the PyHEP notebook fires a long sequence of separate
launches with intermediates — `cupy_multiply`, `cupy_cosh`, `cupy_cos`,
`cupy_subtract` interleaved with `awkward_IndexedArray_getitem_nextcarry_a/b`,
`cupy_take`, `DeviceScanKernel`, each ~0.3–1.2 ms. Collapsing that per-op
launch+materialize chain into one fused cuda.compute kernel over a
`ZipIterator` is the single strongest motivation for the fusion phase.

---

## Objectives (re-scoped)

1. **Finish kernel migration and retire NVRTC.** Not greenfield — close out
   the ~11 remaining CuPy-dispatched kernels and delete the 74 dead `.cu`
   files per the migration doc's Phase 2 batch-2 / Phase 3.
2. **Decide whether fusion is possible, then build it.** Determine if
   cuda.compute can compose multiple ops into one compiled kernel
   (Open Q1 — gating). Only then design the DAG→kernel builder on top of the
   *existing* `_ir.py` operator IR.
3. **Pick one device ABI.** Iterator-composition (incumbent, working) vs.
   `awkward_view_t` struct / reused Numba hashed type (proposed). These
   compete; the struct only earns its keep once fusion emits *custom* kernels.
4. **Extend the backend-neutral lazy core to CPU.** `core.py` already
   anticipates it; add `_connect/cpu/ir_nodes.py` lowering to NumPy/CPU
   kernels. Independent of any NVIDIA library (cuda.compute is GPU-only).
5. **Make error semantics and typetracer interplay first-class**, not
   afterthoughts — both get *worse* under fusion.
6. **Speak the `ak.*` API, not a parallel one (new).** Today lazy execution is
   a bespoke `LazyAwkwardArray` with hand-wired operator overloads and a fixed
   `OpType` set — anything outside that (`ak.zip`, `ak.cartesian`, `ak.num`,
   most of the `ak.*` surface, `np.sqrt(la)`, …) is not expressible lazily and
   never keeps up by hand. Target: lazy arrays participate in awkward's dispatch
   (`__array_ufunc__`, `__array_function__`, the high-level-function machinery)
   so the *existing* `ak.*` code records graph nodes automatically, and fusion
   becomes an optimization pass that fuses the subset it recognizes
   (element-wise + reductions) and leaves the rest as opaque nodes. This inverts
   the current design (a bespoke node per op → general recording + selective
   fusion) and dissolves the "different API" objection.

See "Strategic context" below for why 5 and 6 are now load-bearing rather than
cross-cutting.

---

## Strategic context (dask-awkward is a dead end)

The lazy/fusion work started as a GPU speed optimization, but the surrounding
picture has shifted and the plan must reflect it:

- **dask-awkward is a dead end** — it is pinned to an old dask, and dask will
  not be supported in the way we need. It is therefore *not* a substrate to
  build on and *not* a fallback for the serious runs.
- **The lessons from dask-awkward survive its vehicle.** The two things worth
  emulating — the full `ak.*` API working directly on lazy arrays, and
  typetracer-driven column projection — were never dask's; they are
  awkward-core mechanisms (dispatch, the typetracer) that dask-awkward merely
  consumed. So Objectives 5 and 6 are achievable with dask nowhere in the
  picture.
- **This raises the stakes.** With no dask-awkward to fall back to, a native
  awkward graph/lazy layer is not redundant — it is the *replacement*, and this
  fusion work is a plausible seed for it. That makes the general `ak.*` API
  (Obj 6) and column-projected lazy loading (Obj 5 / Phase 5) *gating for
  analysis*, not nice-to-haves, and it makes **partition scheduling** — the
  piece dask used to provide for free — a real, currently unowned problem.
- **Fusion and lazy loading are synergistic, not competing.** Fusion cuts
  launches and per-op intermediates *within* a partition; typetracing decides
  *what* to load. Because fusion removes the intermediates, each partition's
  peak footprint drops, so larger partitions fit in the same memory — fusion
  actively helps the memory-bound case, it just has to sit behind the
  typetracer/projection front-end (which is not built yet).
- **Packaging (separate library?) — not yet.** The fusion IR looks extractable,
  but the codegens reach into awkward internals (`ak._do.to_buffers`,
  `layout.to_ListOffsetArray64`, the CUDA backend) and would own typetracer
  integration; it is still `_connect`-private and immature. Keep it in-tree
  (or as a first-party companion) until the API stabilizes; spinning it out now
  would ossify an immature interface and force exposing internals.

---

## Re-sequenced tasks

The original six tasks were presented as parallel. They are not: fusion gates
the ABI choice, and error/typetracer design gates fusion. Ordered by
dependency:

### Phase 0 — Close out kernel migration (in flight)
Owned by `cuda-compute-migration-plan.md`. Finish batch-2 (`RegularArray_
combinations_64`, `ListArray_getitem_jagged_apply`, `_shrink`) with device-side
error flags; re-flip the 8 reverts once launch-bound error checks move
device-side; delete the 74 dead `.cu` files and drop their
`populate_kernel_errors` entries. **Gate:** `tests-cuda-kernels` +
`tests-cuda` on GPU.

### Phase 1 — Answer the fusion question — ✅ ANSWERED (2026-07)
Open Q1 resolved: **yes, cuda.compute composes multiple ops into one compiled
kernel — but only through composition we drive, not native multi-primitive
fusion.** There is no deferred graph; each primitive call launches. One kernel
comes from either (a) folding the ops into a single custom op, or (b) composing
`TransformIterator`/`ZipIterator` lazily into a single consuming primitive
(map-into-reduce). So the DAG→single-kernel **emitter is ours to build** — which
Phase 2 did (`_connect/cuda/_fusion_codegen.py`); cuda.compute compiles the
composed op/iterator once and runs it as one kernel.

**Which layer emits it:** the composition layer (our codegen), with
cuda.compute as the compiler/executor. Element-wise chain → 1 `unary_transform`
over a folded op; element-wise+reduction → 1 `segmented_reduce` fed a
`TransformIterator`.

**Measured design finding (matters for the codegen):** fold the whole
expression into ONE op. On A100 the folded op is dead flat vs chain depth
(0.049 ms at depth 1 and 16; 14.9× over separate at depth 16). nsys confirms
the launch counts to the kernel — 9,090 transform kernels, exactly as predicted
if separate launches N/call and folded-op/iterator launch 1/call; `reduce_into`
is a two-pass CUB reduce and the `TransformIterator` map fuses into it (no
separate transform kernel). Stacking one `TransformIterator` per op also fuses
to one launch, but that single kernel is lower quality (slows with nesting)
whereas the folded op is bandwidth-bound and flat — so fold, don't stack.
`_fusion_codegen` already folds; keep it that way.

**Deliverable (done):** full writeup in `phase1-fusion-answer.md`; mechanism
microbenchmark in `probe_fusion_mechanism.py` (separate-primitive vs composed-op
vs iterator, map-only and map+reduce, across depth, with nsys kernel-count
guidance). Corroborated three ways: API semantics (no deferred-graph API);
in-tree proof (`_compute.py::_widen_for_reduce`/`_nonzero_for_reduce` already
fuse map-into-reduce with no intermediate buffer); and the measured Phase 2
benchmark (composed-op time flat vs chain depth = one kernel, ~11–91× on A100).

### Phase 2 — Fusion on the existing IR — ✅ DONE (2026-07)

Built the DAG→kernel step on the operator IR (`_connect/lazy/_ir.py`), kept
separate from `core.py`'s lowering IR:

- **Fusion pass** — `_connect/lazy/_fusion.py`. `fuse(root)` rewrites the graph,
  collapsing maximal element-wise regions (and a single-use element-wise input
  of a reduction → transform+reduce) into `FusedNode`s. Policy is *single-use
  fusion*: fan-out > 1 nodes become shared boundaries (materialize once), so
  CSE of shared sub-expressions falls out of the pass. Reached through
  `IRExecutor.optimize()` (no longer a stub); `compile_and_execute` =
  `execute(optimize(node))`.
- **Backend codegen** — `_connect/cuda/_fusion_codegen.py` emits **one**
  `cuda.compute` kernel per region (`unary_transform` over a `ZipIterator`;
  `TransformIterator` → `segmented_reduce` for the fused reduction).
  `_connect/cpu/_fusion_codegen.py` is the NumPy analogue (one pass over the
  flat content). A codegen miss raises `FusionUnsupported` → eager fallback.
- **Cache-stability constraint met.** Fused ops fold constants into interned
  source (`functools.cache`), so a stable program compiles each kernel once —
  no per-call closure state. Regression-tested
  (`test_fused_op_is_cache_stable`: 1 miss, N-1 hits).
- **No-fuse debug mode.** `compute(fuse=False)` keeps every intermediate
  visible (per-node interpreter); numerically identical to the fused path
  across a test battery.

**Gate — MET.** Fused element-wise **and** transform+reduce beat the
interpreter on the launch-bound benchmark (`studies/cccl/bench_lazy_fusion.py`,
measured della-gpu A100, `bench_lazy_fusion_results.md`):

- Element-wise chains: **~11–91× on GPU** (fused time ~flat vs chain depth =
  the fusion signature; size-independent), ~2–18× on CPU.
- `chain d=4 →sum` (map fused into `segmented_reduce`, one kernel): **10–14×**.
- All device cases `cuda hit ≥ 1` and `match = yes` — the codegen engaged on
  every fusible region and matched the interpreter numerically.

Tests: `tests/test_3792_lazy_fusion.py` (CPU-runnable, incl. GPU-free codegen
surface), `tests-cuda/test_3792_lazy_*` for device.

**Deferred to Phase 5 (not blocking):** *device-side error flags* — the other
Phase-0 constraint. Fused element-wise kernels do no bounds/domain checks and
can't host-sync mid-fusion, so awkward's error semantics under fusion are
handed to Phase 5's error-flag work; until then the eager (`fuse=False`) path
is the debuggable reference.

### Phase 3 — Device ABI — DECIDED (iterator-composition); implementation in `phase3-abi-proposal.md`
**Decision:** adopt **iterator-composition** as the device ABI; **defer
`awkward_view_t`** behind an explicit trigger (a fused/custom kernel needing the
whole nested array as one struct parameter that op-level numba + iterators
cannot express). The Phase 3 gate — a fused kernel consuming an Awkward array
with no new copies, matching the iterator path numerically — is **already met**
by the Phase 2 codegen. Remaining Phase-3 work is therefore not the struct but:
(T1) record the decision, (T2) harden the iterator path into a tested ABI
contract — form coverage matrix, a named+round-tripped output direction
(`cccl_result_to_awkward`), a zero-copy assertion, and a **buffer-provider seam
reserved for Phase 5 column projection**, (T3) a cheap `awkward_view_t`
de-risking spike answering NVIDIA Open Q2, (T4, conditional) the full struct
only if T3 succeeds and the trigger fires. Full task breakdown, acceptance
criteria, and the spike spec: **`studies/cccl/phase3-abi-proposal.md`**.

See "Data layout and the device-view ABI" above for the full argument.
**Phase 2 data point (favours the incumbent):** the
fusion codegen feeds leaves to its single kernel purely by iterator
composition — `ZipIterator` of the SOA columns, `TransformIterator` for
map-into-reduce — with **no struct parameter and no new copies**, and matches
the interpreter numerically on device. So far nothing in fusion demands
`awkward_view_t`; revisit only if a future fused kernel needs the whole nested
array as one argument. Independently of that choice, ✅ **DONE — the recursive
form→iterator builder is available as a supported entry point kept in
`_connect`** (`awkward._connect.cuda.to_cccl_iterator(array, *, dtype=None)`;
`_connect/cuda/__init__.py::to_cccl_iterator`, `__all__`). It returns
`(iterator, metadata)` with a documented metadata contract, degrades with a
clear "install cupy" error when cupy is absent, and is zero-copy for CUDA-backed
input. **Not surfaced as a public `ak.*` name yet** — per review this PoC stays
in `_connect` (the `ak.cpu`/`ak.cuda` namespace exposure in
`awkward/__init__.py` was reverted); promotion to a public surface is deferred
until the lazy/fusion work graduates from PoC (and ideally arrives via the
Objective 6 dispatch route, not a bespoke namespace). External CCCL code (the
PyHEP-talk prototype's hand-rolled buffer extraction) should call the `_connect`
helper instead of reaching into `ak.to_buffers`. Tested in
`tests-cuda/test_3792_public_cccl_iterator.py`. If the struct *is* needed:
**define**
`awkward_view_t` as a struct-of-pointers-to-SOA-columns (not a repackaging of
the current ABI, and not AOS), verify the Numba hashed-type layout/padding
stability, expose a C header, and test three-way interop (Awkward / Numba /
CCCL) against the same device pointer. **Gate:** a fused kernel consuming an
Awkward array with no new copies, matching the iterator path numerically.

### Phase 4 — CPU lazy backend
Add `_connect/cpu/ir_nodes.py` subclassing `core.py`'s `IRNode`, lowering to
NumPy/CPU kernels. Reuses the existing demand-driven/memoized core unchanged.
This is the project's standing "same lazy execution for CPU" goal and does
**not** depend on Phases 1–3. Can run in parallel with Phase 1.

### Phase 5 — Cross-cutting: error flags, typetracer, distribution
- **Error semantics.** Standardize device-side error flags checked through the
  existing `synchronize_cuda` machinery; required by both Phase 0 re-flips and
  Phase 2 fusion.
- **Typetracer / VirtualNDArray (gating for real analysis — promote, see the
  strategic note below).** A single fused kernel needs all input buffers
  resolved upfront — the "which buffers are needed" problem. Reuse **awkward's
  own typetracer** (an awkward-core capability; dask-awkward merely consumed it
  and is not a dependency or a fallback here). This is not optional polish:
  HEP analysis cannot hold the whole event collection in memory, so
  column-projected lazy loading is the *front-end that fusion sits behind* —
  typetrace the graph → minimal column set → project → load a partition → run
  fused kernels. Design the `InputNode` leaf as a form/virtual buffer from the
  start (not a concrete array) so column projection is possible; that leaf
  abstraction is cheap to change now and very expensive later. Four points fix
  *how* this works (a materialized-array shortcut today blocks all of it):

  - **Trace at the buffer level, not the layout level.** The form tells you the
    *shape* of the data, not which *buffers* are read. Column projection is the
    touch-set: run on a typetracer whose buffers are placeholders and record
    which buffer keys are actually dereferenced. Awkward already reports buffer
    touches; reuse it. Knowing the form is necessary but not sufficient.
  - **The blocker is eager materialization in the iterator builder.**
    `awkward_to_cccl_iterator` currently calls `ak._do.to_buffers(layout)` and
    wraps every leaf in `cp.asarray`, so *building* the iterator materializes
    every buffer it is handed — before `.compute` and regardless of what the
    fused op reads. A `VirtualNDArray` leaf would be forced to materialize at
    build time, defeating projection. **Refactor:** the builder must take a
    *form + a lazy buffer provider* and bind device buffers only for the
    projected keys, instead of pulling `to_buffers` up front. As it stands this
    is a fully-materialized-array optimization.
  - **The form↔iterator homomorphism is the lever, not an obstacle.** The
    builder is structural — `NumpyArray↔flat buffer`, `RecordArray↔ZipIterator`,
    `IndexedArray↔PermutationIterator`, `ListOffsetArray↔content-iter +
    offsets`. Because each iterator leaf maps 1:1 to a form node, the iterator
    skeleton can be built from the form and real buffers bound only for the kept
    columns; give each leaf its buffer-key provenance so the skeleton knows
    which buffer it *would* need.
  - **Compute the needed-set at plan time, not by post-hoc iterator reporting.**
    needed-set = (structural columns read straight off the fused IR: the fused
    op source literally names its columns, e.g. `(t[0]*2)+1`, so a pure-fusion
    pipeline needs *no* typetracer) ∪ (buffer-level typetracer touch-set for the
    opaque `ak.*` nodes from dispatch compatibility, whose reads aren't visible
    structurally). Determining this before building iterators means the unneeded
    buffers are never materialized; discovering it *from* the iterator at
    `.compute` only reports the waste after paying for it.
- **Partition scheduling (newly unowned — no dask-awkward).** Distributing the
  IR is *more* than serialization. With dask-awkward a dead end (pinned to an
  old dask; dask will not be supported the way we need), there is no scheduler
  to lean on, so chunking the collection into partitions, ordering the DAG
  across partitions, parallelism, and spilling are all **our problem** and
  currently have no owner. The graph itself is already inspectable
  (`walk` / `topological_order`); the serialization piece is easy (`InputNode`
  detach/reattach to form + buffer handles). The scheduler is the hard,
  expensive, load-bearing piece — call it out explicitly rather than assume
  dask-style infrastructure exists.

### Phase 6 — NVIDIA sync (continuous)
Align on the open questions below; keep tracking the linked issues.

### Phase 7 — `ak.*` dispatch compatibility (high priority; parallel to 3–5)
Make lazy arrays honor awkward's dispatch so the existing `ak.*` surface and
NumPy ufuncs record graph nodes automatically, instead of the fixed
`LazyAwkwardArray` op set. Gating for adoption (Objective 6, "Strategic
context") and a prerequisite for a lazy layer that could replace dask-awkward's
role. Sketch:

- Route `__array_ufunc__` / `__array_function__` and awkward's high-level
  `@high_level_function` dispatch into node construction, so `ak.sum(la)`,
  `np.sqrt(la)`, `ak.combinations(la, 2)` build IR rather than erroring.
- Generalize the IR to record *any* awkward op as a node (opaque by default);
  `fuse` then only fuses the subset it recognizes (element-wise + reductions)
  and leaves opaque nodes as region boundaries — exactly the current
  boundary-vs-interior split, widened.
- Keep `compute(fuse=…)` and the eager fallback semantics; an opaque node
  simply dispatches to its eager `ak.*` implementation.
- **Gate:** an analysis snippet written in ordinary `ak.*` (e.g. the dimuon
  pipeline) runs unchanged on a lazy array and fuses its element-wise regions.

---

## Open questions (pruned — several were already answered in-tree)

Still genuinely for NVIDIA:
1. ✅ **RESOLVED (was gating)** — mechanism to fuse multiple cuda.compute
   expressions into one compiled kernel: **build-our-own** on top of
   op-/iterator-composition; there is no native deferred-graph fuser (each
   primitive launches). See `phase1-fusion-answer.md`. Remaining NVIDIA angle:
   *would a future cuda.compute gain a first-class deferred-graph fuser?* If so
   the codegen could target it instead of hand-composing — nice-to-have, not
   blocking.
2. Can CCCL consume a device struct (`awkward_view_t`) directly, and is there
   a supported path to feed Numba-lowered hashed types into CCCL-compiled
   kernels? (Only pursued if Phase 3 chooses the struct.)
3. Is there any NVIDIA-supported CPU⇄GPU dispatch layer, or is CPU strictly
   our own path? (Working assumption: our own — Phase 4.)

Reclassified — **not** open questions, already handled in-repo:
- *"Access the compute graph without executing"* → yes; the IR is inspectable
  today (`walk` / `topological_order`). Real work is serialization (Phase 5).
- *"Debugging intermediate arrays"* → the interpreter already keeps
  intermediates visible; the actual issue is that **fusion destroys that
  visibility**. Design a debug/no-fuse mode in Phase 2.
- *"Delayed execution for CPU"* → the lazy core is already backend-neutral;
  Phase 4, not a question.
- *"Interplay with VirtualNDArray"* → reuse awkward's own typetracer
  (Phase 5), not a novel problem — but not free either (see the strategic note:
  it is the gating front-end, not an afterthought).

---

## Success metrics (new — the original had none)

Import the discipline from `cuda-compute-migration-plan.md`:
- Every phase reports ms/op on della at ≥2 sizes (200k and 2M lists), against
  the interpreter/raw-kernel baseline.
- Fusion is justified only where it amortizes launch overhead — target the
  **launch-bound kernels the migration doc flagged** (their raw time barely
  grows over 10× data). Bandwidth-bound kernels are not fusion wins.
- Revert criterion, mirroring Phase 1 of the migration: any fused path that
  fails to beat the interpreter at 2M lists, or that JITs per call, is
  reverted and documented.

## Risks and invariants (carried forward)

- **Error semantics:** legacy message text must be preserved (generated
  `tests-cuda-kernels` match on it); fusion forces device-side flags.
- **Cache-stable compilation:** ops must not close over per-call state
  (measured 1.8 s/call regression otherwise).
- **Dtype coverage & stream correctness:** unchanged from the migration doc.
- **ABI drift:** if the hashed struct is adopted, its layout/padding must be
  pinned and tested, or Numba/CCCL interop breaks silently.

---

## Benchmarking and profiling

Three distinct jobs — do not conflate them. `tests-cuda-kernels` is a
**correctness** gate (6968 generated cases from `kernel-test-data.json`, tiny
inputs), not a benchmark; its ~20 min is dominated by per-test host dispatch +
the one-time `initialize_cuda_kernels()` NVRTC compile, not GPU compute. Never
use it as the iteration loop or the timing signal.

**Cutting the correctness loop.** `pytest-xdist` is already in the test
requirements; the suite is host/launch-bound, so parallelize:

```bash
python -m pytest tests-cuda-kernels -n auto -q        # or -n 8
```

Returns flatten past ~4–8 workers (each worker re-pays the NVRTC compile at
startup and all share one GPU) — measure `-n 4` vs `-n 8` on della once. For
iteration, select instead of running all 349 files:

```bash
python -m pytest tests-cuda-kernels/test_cudaawkward_ListArray_getitem_next_range_64.py -q
python -m pytest tests-cuda-kernels -k "ListArray_getitem" -q
python -m pytest tests-cuda-kernels --lf -x -q        # last-failed, fail fast
```

Run the full `-n auto` sweep only as the final gate.

**Benchmarking (timing).** Use the in-repo harness, not an ad-hoc script:
`studies/cccl/bench_flip17.py` times at the **operation level** (works
identically on both sides of a raw→compute flip) and writes JSON for
before/after compare:

```bash
python studies/cccl/bench_flip17.py --out before.json   # pre-change commit
python studies/cccl/bench_flip17.py --out after.json    # post-change commit
python studies/cccl/bench_flip17.py --compare before.json after.json
```

Non-negotiable GPU-timing pattern — warm up, then time device work with CUDA
events (wall-clock is meaningless without a sync):

```python
import cupy as cp
op = lambda: ak.to_regular(arr)          # op under test
op(); cp.cuda.Device().synchronize()     # warmup: NVRTC + numba/cuda.compute JIT + op-cache fill
start, end = cp.cuda.Event(), cp.cuda.Event()
start.record()
for _ in range(100):
    op()
end.record(); end.synchronize()
print(cp.cuda.get_elapsed_time(start, end) / 100, "ms/op")
```

Two gotchas already learned in `cuda-compute-migration-plan.md`: (1) always
discard the first call — a cold call measures compilation; if the *second* call
isn't fast, the op is closing over per-call state and defeating the
cuda.compute op cache (the ~1.8 s/call regression), which is a bug in the impl,
not the kernel. (2) Time `initialize_cuda_kernels()` separately (one-time
startup); don't fold it into per-op numbers.

**Profiling (where the time goes).** Three tools, three questions:

- **Device timeline / launch overhead** — `nsys`, wired up in
  `studies/cccl/profile_cccl.py`:
  `nsys profile -t cuda,nvtx python studies/cccl/profile_cccl.py`. Bracket hot
  regions with `nvtx.annotate("name")` +
  `--capture-range=nvtx --nvtx-capture="name"`; view via nsightful/Perfetto or
  `nsys stats`. This is what surfaces the per-op launch chain that motivates
  fusion.
- **Per-kernel counters (occupancy, achieved bandwidth)** — Nsight Compute:
  `ncu --set full -k "regex:awkward_" -c 20 python script.py`. Use it to
  classify launch-bound vs bandwidth-bound (the migration doc's revert
  criterion).
- **Host-side Python overhead** — `cProfile` (also in `profile_cccl.py`); for
  launch-bound kernels the dispatch cost dominates and is the real lever.

Rule of thumb: run `nsys` first to see whether you're launch-bound (many small
kernels + gaps) or compute/bandwidth-bound (few long kernels); only then reach
for `ncu` or `cProfile`.
