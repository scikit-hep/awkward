# Phase 3 — device ABI: review and implementation proposal

Status: proposal · 2026-07. Read with the "Data layout and the device-view ABI"
section and Phase 3 of `cccl-followup-plan-revised.md`.

---

## 1. Review — where Phase 3 actually stands

**The question.** Phase 3 was framed as a choice between two device-view
substrates:

- **cuda.compute view** — `helpers.py::awkward_to_cccl_iterator` builds composed
  SOA iterators (`NumpyArray→flat buffer`, `RecordArray→ZipIterator`,
  `IndexedArray→PermutationIterator`, `ListOffsetArray→content-iter + offsets`),
  consumed by CUB/Thrust primitives. This is what the fusion codegen uses.
- **Numba CUDA view** — `arrayview_cuda.py` passes `(pos, start, stop,
  arrayptrs, pylookup)`, where `arrayptrs` is a *device array of pointers* to
  the individual buffers, consumed by `numba.cuda.jit` kernels.

`awkward_view_t` is the proposal to unify these into one C
struct-of-pointers-to-SOA-columns that both a numba kernel and a CCCL kernel
could consume.

**The decision is effectively already made by Phase 2 evidence.**

- The fusion codegen consumes leaves *purely by iterator composition*
  (`ZipIterator` of the SOA columns, `TransformIterator` for map-into-reduce),
  with **no struct parameter and no new copies**, and matches the interpreter
  numerically on an A100. The Phase 3 gate — "a fused kernel consuming an
  Awkward array with no new copies, matching the iterator path numerically" —
  is therefore **already met by the incumbent**.
- The incumbent *already bridges numba and CCCL at the op level*: the fused op
  is a numba-compiled Python function handed to a cuda.compute primitive that
  iterates via CCCL iterators. So numba↔CCCL interop already works, at the
  granularity fusion needs, **without any struct**.
- `awkward_to_cccl_iterator` is wrapped as a supported *internal* entry point
  (`awkward._connect.cuda.to_cccl_iterator`, tested). Per review it is **not**
  surfaced as a public `ak.*` name while this is a PoC — the `ak.cpu`/`ak.cuda`
  exposure was reverted; a public surface is deferred (ideally via the
  Objective 6 dispatch route, not a bespoke namespace).

**Conclusion.** Phase 3 is not "build `awkward_view_t`." It is: (1) record the
decision, (2) harden the iterator path into a first-class, tested ABI contract,
(3) cheaply de-risk the struct so the deferral is *informed*, not blind. Build
`awkward_view_t` only if a concrete trigger fires.

---

## 2. Recommendation

**Adopt iterator-composition as the device ABI. Defer `awkward_view_t`** behind
an explicit trigger:

> Build `awkward_view_t` only when a fused/custom kernel must receive the *whole
> nested array as a single kernel parameter* — e.g. a hand-written CUDA kernel
> or a `numba.cuda.jit` kernel that navigates the structure itself — **and**
> op-level numba functions + iterator composition cannot express it.

Until then the struct is unjustified cost: defining it, pinning the Numba
hashed-type layout/padding, exposing a C header, and maintaining three-way
interop, all to duplicate what iterator composition already does zero-copy.

Rationale is the SOA argument from the plan: Awkward is struct-of-arrays end to
end, so `ZipIterator` is the natural zero-copy fit; a packed struct buys nothing
for iteration and an AOS `gpu_struct` would *cost* a copy.

---

## 3. Implementation tasks

### T1 — Decision record (small, now)

Write an ADR (this doc is the seed): iterator-composition is the ABI;
`awkward_view_t` deferred; the trigger above; a named owner for revisiting.
Update plan Phase 3 to "decided," pointing here.
**Done when:** ADR merged and Phase 3 status flipped.

### T2 — Harden the iterator ABI into a tested contract (the real Phase-3 work)

The iterator path works but is under-specified as an *interface*. Make it
first-class:

1. **Coverage matrix — document and test the supported set, one case per form:**

   | Form | ABI behavior |
   |------|--------------|
   | `NumpyForm` | flat CuPy buffer |
   | `RecordForm` | `ZipIterator(*fields)` |
   | `IndexedForm` over leaf | `PermutationIterator(content, index)` |
   | `IndexedForm` over a list | `NotImplementedError` (project first) |
   | `ListOffsetForm` | content iterator + offsets in metadata |
   | top-level `ListForm`/`RegularForm` | normalized to `ListOffset` (`validate_iterator_layout`) |
   | nested `ListForm` / `RegularForm` | `NotImplementedError` (non-contiguous / not yet supported) |
   | `IndexedOptionForm` (missing values) | `NotImplementedError` |
   | `UnionForm`, `ByteMasked`/`BitMasked`/`Unmasked` | `NotImplementedError` |

   GPU-gated tests asserting each supported form round-trips and each
   unsupported form raises a *clear* `NotImplementedError` (no silent wrong /
   OOB — the same discipline as the fusion codegen). This is the ABI's
   "what does it accept" contract, currently only implicit in the code.

2. **Name the output direction.** The reverse — result buffer + offsets →
   `ak.Array` — is currently ad hoc inside `_fusion_codegen` (build a
   `ListOffsetArray` from `out_content` + offsets). Extract a documented
   `cccl_result_to_awkward(out_buffer, metadata)` so both directions of the ABI
   are named and symmetric. **Round-trip test:** `layout → iterator → identity
   transform → cccl_result_to_awkward → layout` equals the original.

3. **Lock zero-copy with a test, not a claim.** For an array already on the
   device, assert that building the iterator allocates *no* new device memory
   for the input buffers (CuPy mempool `used_bytes` delta ≈ 0, only offsets/
   output allocate). Zero-copy on the input side is the ABI's entire reason to
   exist; guard it.

4. **Reserve the Phase-5 projection seam now.** Give `awkward_to_cccl_iterator`
   (and the public wrapper) an optional *buffer-provider* / projected-key
   parameter — a callable `key -> device buffer` instead of the eager
   `ak._do.to_buffers` + `cp.asarray`. Unused today (pass a provider that just
   returns the materialized buffer), but it is the exact seam the Phase 5
   "form + lazy buffer provider" column-projection refactor plugs into. Design
   it in now so the ABI is not re-cut when lazy loading lands. **Cross-ref:
   Phase 5.**

**Done when:** coverage matrix tested, `cccl_result_to_awkward` named + round-
trip-tested, zero-copy asserted, buffer-provider parameter in the signature and
documented.

### T3 — De-risk `awkward_view_t` with a minimal spike (cheap; answers Open Q2)

Before any commitment, answer NVIDIA Open Q2 ("can CCCL consume a device struct,
and can Numba-lowered hashed types feed CCCL-compiled kernels?") empirically.
This is a *probe*, in the spirit of `probe_fusion_mechanism.py` — not production.

**Experiment.** Two SOA columns `x`, `y` (separate device buffers) and a length,
exposed as a single struct-of-pointers `{const T* x; const T* y; int64_t n;}`:

1. **Numba side:** a `numba.cuda.jit` kernel that receives the pointer table
   (as the existing `arrayptrs` device array is passed) and reads `x[i]`, `y[i]`
   via `numba.carray`. Confirm it computes `x[i]+y[i]` correctly. (This mirrors
   `arrayview_cuda.py`'s pointer-table mechanism, minus the pos-tree.)
2. **CCCL side:** feed the *same* device buffers to a cuda.compute primitive two
   ways and compare: (a) as a `ZipIterator(x, y)` (the incumbent), and (b), if
   the API allows, wrapped in a `gpu_struct`/pointer-carrying op — checking
   whether a cuda.compute op can dereference struct-carried pointers at all.
3. **Same-pointer, no-copy:** assert both sides read the identical device
   addresses (no staging copy) and produce identical results.

**Pass/fail:** report yes/no for "a single struct-of-pointers is consumable by
both a numba kernel and a cuda.compute op against the same buffers," plus any
Numba hashed-type layout/padding surprises. A "no" *confirms* the deferral (stick
with iterators); a "yes" tells us the escape hatch exists if the trigger fires.

**Done when:** the probe runs on della and the yes/no + notes are recorded here
and carried to Phase 6 (NVIDIA sync).

### T4 — Full `awkward_view_t` (conditional — only if T3 = yes AND trigger fires)

Define `awkward_view_t` as a struct-of-pointers-to-SOA-columns (not AOS, not a
repackaging of the current N-separate-pointers ABI; list layouts carry
`starts`/`stops` or `offsets`, never `parents`). Pin the Numba hashed-type
layout/padding with a test, expose a C header, and test three-way interop
(Awkward buffers / numba kernel / CCCL kernel) against one device pointer,
matching the iterator path numerically. **Explicitly not started until
triggered.**

---

## 4. Sequencing, dependencies, risks

- **Now:** T1 (record) + T2 (harden) — T2 is the actual Phase-3 completion.
  T3 is a cheap parallel spike that also feeds Phase 6.
- **Deferred:** T4, behind the trigger.
- **Phase 5 coupling (important):** T2's buffer-provider seam is where the
  typetracer/column-projection front-end plugs in. If T2 ships without it, the
  ABI gets re-cut when lazy loading lands — so reserve the parameter even though
  Phase 5 fills it in later. The ABI (how buffers reach the kernel) and
  projection (which buffers) meet exactly here.
- **Risk — over-building:** the temptation is to build `awkward_view_t` because
  it's the "interesting" artifact. The Phase 2 evidence says it earns nothing
  until a custom-kernel need appears; T3 keeps the option open cheaply without
  paying for it.
- **Risk — ABI drift (carried from the plan):** if T4 ever happens, the hashed
  struct's layout/padding must be pinned and tested, or Numba/CCCL interop
  breaks silently.

## 5. Phase-3 acceptance (revised)

Phase 3 is **done** when: the ADR is recorded (T1); the iterator ABI is hardened
with a tested coverage matrix, a named + round-trip-tested output direction, a
zero-copy assertion, and the projection seam reserved (T2); and the struct spike
is answered (T3). `awkward_view_t` (T4) is out of scope unless its trigger fires.
