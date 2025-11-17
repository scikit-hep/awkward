import awkward as ak
import numpy as np
import cupy as cp
import math
from cuda.compute import (
    ZipIterator, PermutationIterator, CountingIterator, TransformIterator,
    binary_transform, unary_transform, gpu_struct, three_way_partition, segmented_reduce, OpKind,
    exclusive_scan
)


# 3 events with different numbers of electrons and muons in each
# (numerical values are made up and aren't physically meaningful)
# ---
# legend:
# - 'pt': transverse momentum
# - 'eta': pseudorapidity (collider detector angle, e.g. CMS detector)
# - 'phi': azimuthal angle (collider detector angle, e.g. CMS detector)
electrons = ak.Array(
    [
        # 2 electrons
        {"pt": [50.0, 60.0], "eta": [2.1, 2.2], "phi": [0.6, 0.7]},
        # 1 electron
        {"pt": [30.0], "eta": [-1.5], "phi": [0.3]},
        # 0 electrons
        {"pt": [], "eta": [], "phi": []},
    ]
)
muons = ak.Array(
    [
        # 1 muon
        {"pt": [45.0], "eta": [2.5], "phi": [0.4]},
        # 2 muons
        {"pt": [25.0, 35.0], "eta": [-2.0, 1.0], "phi": [0.5, 0.6]},
        # 1 muon
        {"pt": [15.0], "eta": [0.0], "phi": [0.7]},
    ]
)

events = ak.zip({"electrons": electrons, "muons": muons}, depth_limit=1)


@gpu_struct
class Particle:
    pt: np.float32
    eta: np.float32
    phi: np.float32


def physics_analysis(events: ak.Array) -> ak.Array:
    """
    A oversimplified physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass.
    """
    # select only electrons with pt > 40
    selected_electrons = events.electrons[events.electrons.pt > 40.0]
    # select only muons with pt > 20 and abs(eta) < 2.4
    selected_muons = events.muons[
        (events.muons.pt > 20.0) & (abs(events.muons.eta) < 2.4)
    ]

    # choose exactly 2 leptons (electrons or muons)
    two_electrons = selected_electrons[ak.num(
        selected_electrons.pt, axis=-1) == 2]
    two_muons = selected_muons[ak.num(selected_muons.pt, axis=-1) == 2]

    return ak.zip(
        {
            "electron": invariant_mass(two_electrons),
            "muon": invariant_mass(two_muons),
        }
    )


def invariant_mass(two_particles: ak.Array) -> ak.Array:
    """Compute invariant mass of two particles given their pt, eta, phi."""
    pt1, eta1, phi1 = (
        two_particles[:, 0].pt,
        two_particles[:, 0].eta,
        two_particles[:, 0].phi,
    )
    pt2, eta2, phi2 = (
        two_particles[:, 1].pt,
        two_particles[:, 1].eta,
        two_particles[:, 1].phi,
    )

    # https://en.wikipedia.org/wiki/Invariant_mass#Collider_experiments
    m2 = 2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
    return np.sqrt(m2)


def create_list_item_iterators_contiguous(base_iterator, item_indices, num_events, items_per_event):
    """
    Create iterators for accessing specific items within each list.

    This version assumes items are stored contiguously: event i's items are at positions
    [i*items_per_event, i*items_per_event+1, ..., i*items_per_event+items_per_event-1]

    Args:
        base_iterator: CCCL iterator over the flattened content
        item_indices: List of item positions to access (e.g., [0, 1] for first two items)
        num_events: Number of lists/events
        items_per_event: Number of items in each list (e.g., 2 for particle pairs)

    Returns:
        List of PermutationIterators, one for each item index

    Note: Currently materializes small index arrays (num_events elements). For very large
    datasets, this could be optimized using TransformIterator with CUDA device functions.
    """
    result_iterators = []

    for item_idx in item_indices:
        # Create indices: [item_idx, item_idx+items_per_event, item_idx+2*items_per_event, ...]
        # For item_idx=0, items_per_event=2: [0, 2, 4, ...]
        # For item_idx=1, items_per_event=2: [1, 3, 5, ...]
        indices = cp.arange(item_idx, item_idx + num_events *
                            items_per_event, items_per_event, dtype=np.int64)

        # Create PermutationIterator using these indices
        perm_iter = PermutationIterator(base_iterator, indices)
        result_iterators.append(perm_iter)

    return result_iterators


def awkward_to_cccl_iterator(array=None, form=None, buffers=None, dtype=None, return_offsets=True):
    """
    Convert an Awkward Array to a CCCL iterator (zero-copy).

    This function recursively traverses the Awkward form structure and constructs
    the corresponding CCCL iterator:
    - NumpyArray -> CuPy buffer
    - RecordArray -> ZipIterator over field iterators
    - IndexedArray -> PermutationIterator with index buffer
    - ListOffsetArray -> Iterator for the flattened content

    The result is a CCCL iterator that provides zero-copy access to GPU buffers,
    with automatic handling of indexing, records, and jagged array flattening.

    Example usage:
        # Convert an Awkward Array with structure: IndexedArray -> RecordArray -> ...
        iterator, metadata = awkward_to_cccl_iterator(my_array, dtype=np.float32)
        # metadata = {"form": ..., "buffers": ..., "offsets": ..., "length": ...}
        # Now `iterator` can be used directly with CCCL algorithms
        # For application-specific indexing, wrap with additional PermutationIterators

    Args:
        array: Awkward Array (if starting fresh)
        form: Awkward form (from ak.to_buffers)
        buffers: Buffer dict (from ak.to_buffers)
        dtype: Optional dtype to cast to (e.g., np.float32 for GPU structs)
        return_offsets: If True, extract and return offsets for list structures (default: True)

    Returns:
        CCCL iterator or CuPy array representing the structure
        Dictionary with metadata: {"form": ..., "buffers": ..., "offsets": ..., "length": ...}
        - offsets: CuPy array of offsets if list structure exists, else None
        - length: Array length from ak.to_buffers
    """
    # Initial call: extract form and buffers from array
    initial_call = form is None and buffers is None
    length = None

    if initial_call:
        if array is None:
            raise ValueError(
                "Must provide either 'array' or both 'form' and 'buffers'")
        array_gpu = ak.to_backend(array, "cuda")
        form, length, buffers = ak.to_buffers(array_gpu)

    # Helper to extract offsets from the form structure
    def extract_offsets_from_form(form_to_search):
        """Navigate through form structure to find and extract list offsets."""
        search_form = form_to_search

        # Unwrap IndexedForm/IndexedOptionForm
        if isinstance(search_form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm)):
            search_form = search_form.content

        # Unwrap RecordForm (use first field)
        if isinstance(search_form, ak.forms.RecordForm):
            search_form = search_form.contents[0]

        # Extract offsets from ListOffsetForm/ListForm
        if isinstance(search_form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
            offsets_key = f"{search_form.form_key}-offsets"
            return cp.asarray(buffers[offsets_key], dtype=np.int64)

        return None

    # Extract offsets if this is the initial call and offsets are requested
    offsets = None
    if initial_call and return_offsets:
        offsets = extract_offsets_from_form(form)

    # Helper to create metadata dict for return
    def make_metadata():
        return {
            "form": form,
            "buffers": buffers,
            "offsets": offsets,
            "length": length
        }

    # Base case: NumpyArray - return the flat data buffer
    if isinstance(form, ak.forms.NumpyForm):
        data_key = f"{form.form_key}-data"
        buffer = buffers[data_key]
        if dtype is not None:
            buffer = cp.asarray(buffer, dtype=dtype)
        else:
            buffer = cp.asarray(buffer)

        if initial_call:
            return buffer, make_metadata()
        else:
            return buffer, (form, buffers)

    # RecordArray: create ZipIterator over all fields
    elif isinstance(form, ak.forms.RecordForm):
        field_iterators = []
        for field_form in form.contents:
            field_iter, _ = awkward_to_cccl_iterator(
                form=field_form, buffers=buffers, dtype=dtype, return_offsets=False
            )
            field_iterators.append(field_iter)

        result = ZipIterator(*field_iterators)
        if initial_call:
            return result, make_metadata()
        else:
            return result, (form, buffers)

    # IndexedArray: create PermutationIterator with index mapping
    elif isinstance(form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm)):
        index_key = f"{form.form_key}-index"
        index_buffer = cp.asarray(buffers[index_key])

        # Recursively get iterator for the content
        content_iter, _ = awkward_to_cccl_iterator(
            form=form.content, buffers=buffers, dtype=dtype, return_offsets=False
        )

        result = PermutationIterator(content_iter, index_buffer)
        if initial_call:
            return result, make_metadata()
        else:
            return result, (form, buffers)

    # ListOffsetArray: return iterator for the flattened content
    # (Offsets are extracted at the top level if this is an initial call)
    elif isinstance(form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
        # Recursively handle the content (which is already flattened)
        content_iter, _ = awkward_to_cccl_iterator(
            form=form.content, buffers=buffers, dtype=dtype, return_offsets=False
        )

        if initial_call:
            return content_iter, make_metadata()
        else:
            return content_iter, (form, buffers)

    else:
        raise NotImplementedError(
            f"Form type {type(form).__name__} not yet supported. "
            f"Please add support or file an issue."
        )


def segmented_reduce_helper(input_iter, offsets, op, init_value, num_segments):
    """
    Helper function for segmented reduce following the Schwartz pattern.

    Wraps the common pattern of splitting offsets into start/end for segmented operations.

    Args:
        input_iter: Input iterator to reduce
        offsets: Offsets array defining segment boundaries
        op: Reduction operator (OpKind or callable)
        init_value: Initial value for the reduction (numpy scalar)
        num_segments: Number of segments

    Returns:
        CuPy array containing one reduced value per segment
    """
    output = cp.empty(num_segments, dtype=init_value.dtype)
    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    segmented_reduce(
        input_iter,
        output,
        start_offsets,
        end_offsets,
        op,
        init_value,
        num_segments=num_segments
    )
    return output


def process_leptons_cccl(leptons: ak.Array, pt_min: float, eta_max: float = None, lepton_name: str = "lepton") -> ak.Array:
    """
    Process leptons using CCCL primitives with TransformIterator and PermutationIterator.

    Key CCCL operations demonstrated:
    1. TransformIterator - for lazy filtering (mask computation)
    2. segmented_reduce - for counting passing particles per event
    3. PermutationIterator - for accessing filtered data
    4. binary_transform - for invariant mass computation

    Args:
        leptons: Awkward Array of leptons with pt, eta, phi fields
        pt_min: Minimum pt cut
        eta_max: Maximum |eta| cut (None to skip)
        lepton_name: Name for debugging

    Returns:
        Awkward Array of invariant masses for events with exactly 2 passing leptons
    """
    # Move to GPU (zero-copy)
    leptons_gpu = ak.to_backend(leptons, "cuda")

    # Get iterators using our general helper (zero-copy!)
    # The helper now automatically extracts offsets for list structures!
    base_iter, metadata = awkward_to_cccl_iterator(
        leptons_gpu, dtype=np.float32)

    # Extract metadata
    offsets = metadata["offsets"]
    num_events = metadata["length"]
    total_particles = int(offsets[-1])

    # Step 1: Create mask iterator using TransformIterator (lazy, no materialization!)
    # Note: ZipIterator returns values where fields are accessed by index:
    # index 0 = pt, index 1 = eta, index 2 = phi
    def make_mask_fn(pt_min_val, eta_max_val):
        if eta_max_val is not None:
            def mask_fn(p):
                pt = p[0]
                eta = p[1]
                return np.int32(1) if (pt > pt_min_val and abs(eta) < eta_max_val) else np.int32(0)
        else:
            def mask_fn(p):
                pt = p[0]
                return np.int32(1) if pt > pt_min_val else np.int32(0)
        return mask_fn

    mask_iter = TransformIterator(base_iter, make_mask_fn(pt_min, eta_max))

    # Step 2: Count passing particles per event using segmented_reduce helper
    event_counts = segmented_reduce_helper(
        mask_iter, offsets, OpKind.PLUS, np.array(
            0, dtype=np.int32), num_events
    )

    # Step 3: Find events with exactly 2 passing particles
    events_with_two = cp.where(event_counts == 2)[0]

    if len(events_with_two) == 0:
        return ak.Array([])

    # Step 4: Build indices for the two particles in each good event
    # Materialize mask once using unary_transform (needed for index extraction)
    mask_array = cp.empty(total_particles, dtype=np.int32)

    def identity(x):
        return x
    unary_transform(mask_iter, mask_array, identity, total_particles)

    # Extract particle indices for good events
    # TODO: This could potentially be optimized further with a custom CCCL kernel
    # that directly computes indices without the Python loop
    particle1_indices = cp.empty(len(events_with_two), dtype=np.int32)
    particle2_indices = cp.empty(len(events_with_two), dtype=np.int32)

    for idx, event_idx in enumerate(events_with_two):
        event_start = int(offsets[event_idx])
        event_end = int(offsets[event_idx + 1])
        event_mask = mask_array[event_start:event_end]
        passing_indices = cp.where(event_mask == 1)[0] + event_start
        particle1_indices[idx] = passing_indices[0]
        particle2_indices[idx] = passing_indices[1]

    # Step 5: Create PermutationIterators for the two particles (zero-copy iteration!)
    particle1_iter = PermutationIterator(base_iter, particle1_indices)
    particle2_iter = PermutationIterator(base_iter, particle2_indices)

    # Step 6: Compute invariant mass using binary_transform (CCCL!)
    output = cp.empty(len(events_with_two), dtype=np.float32)

    # Note: ZipIterator returns values where fields are accessed by index:
    # index 0 = pt, index 1 = eta, index 2 = phi
    def invariant_mass_op(p1, p2) -> np.float32:
        pt1, eta1, phi1 = p1[0], p1[1], p1[2]
        pt2, eta2, phi2 = p2[0], p2[1], p2[2]

        # Use the simplified invariant mass formula (same as original code)
        # https://en.wikipedia.org/wiki/Invariant_mass#Collider_experiments
        m2 = 2 * pt1 * pt2 * (math.cosh(eta1 - eta2) - math.cos(phi1 - phi2))
        return math.sqrt(m2)

    binary_transform(particle1_iter, particle2_iter, output,
                     invariant_mass_op, len(events_with_two))

    # Return as Awkward Array (wrapping the CuPy result)
    return ak.Array(output)


def physics_analysis_cccl(events: ak.Array) -> ak.Array:
    """
    CCCL-based physics analysis selecting events with exactly 2 leptons (electrons or muons)
    and computing their invariant mass using cuda.compute primitives.
    """
    # Move data to GPU (zero-copy)
    events_gpu = ak.to_backend(events, "cuda")

    # Process electrons: filter (pt > 40), select exactly 2, compute mass with CCCL
    electron_masses = process_leptons_cccl(
        events_gpu.electrons,
        pt_min=40.0,
        eta_max=None,  # No eta cut for electrons in this example
        lepton_name="electron"
    )

    # Process muons: filter (pt > 20, |eta| < 2.4), select exactly 2, compute mass with CCCL
    muon_masses = process_leptons_cccl(
        events_gpu.muons,
        pt_min=20.0,
        eta_max=2.4,
        lepton_name="muon"
    )

    return ak.zip({"electron": electron_masses, "muon": muon_masses})


if __name__ == "__main__":
    # ipython -i studies/cccl/playground.py to play around with `events` and `physics_analysis`

    # Run original function
    inv_mass = physics_analysis(events)
    print("Original physics_analysis() results:")
    print("  Electron invariant masses (in GeV):", inv_mass.electron)
    print("  Muon invariant masses (in GeV):", inv_mass.muon)

    # Run CCCL-based function
    inv_mass_cccl = physics_analysis_cccl(events)
    print("\nCCCL physics_analysis_cccl() results:")
    print("  Electron invariant masses (in GeV):", inv_mass_cccl.electron)
    print("  Muon invariant masses (in GeV):", inv_mass_cccl.muon)

    # Verify results match
    print("\n" + "="*60)
    print("VERIFICATION: Comparing results...")
    print("="*60)

    # Convert CCCL results back to CPU for comparison
    inv_mass_cccl_cpu = ak.to_backend(inv_mass_cccl, "cpu")

    # Helper to convert to numpy safely
    def to_flat_numpy(arr):
        arr_np = ak.to_numpy(arr)
        return arr_np.flatten() if arr_np.ndim > 1 else arr_np

    # Compare electron masses
    if len(inv_mass.electron) == 0 and len(inv_mass_cccl_cpu.electron) == 0:
        electron_match = True
        print("✓ Electron masses: Both empty (MATCH)")
    elif len(inv_mass.electron) > 0 and len(inv_mass_cccl_cpu.electron) > 0:
        orig_vals = to_flat_numpy(inv_mass.electron)
        cccl_vals = to_flat_numpy(inv_mass_cccl_cpu.electron)
        electron_match = np.allclose(
            orig_vals, cccl_vals, rtol=1e-5, atol=1e-6)
        if electron_match:
            print("✓ Electron masses: MATCH (within tolerance)")
        else:
            print("✗ Electron masses: MISMATCH")
            print(f"  Original: {orig_vals}")
            print(f"  CCCL:     {cccl_vals}")
    else:
        electron_match = False
        print("✗ Electron masses: MISMATCH (different lengths)")

    # Compare muon masses
    if len(inv_mass.muon) == 0 and len(inv_mass_cccl_cpu.muon) == 0:
        muon_match = True
        print("✓ Muon masses: Both empty (MATCH)")
    elif len(inv_mass.muon) > 0 and len(inv_mass_cccl_cpu.muon) > 0:
        orig_vals = to_flat_numpy(inv_mass.muon)
        cccl_vals = to_flat_numpy(inv_mass_cccl_cpu.muon)
        muon_match = np.allclose(orig_vals, cccl_vals, rtol=1e-5, atol=1e-6)
        if muon_match:
            print("✓ Muon masses: MATCH (within tolerance)")
        else:
            print("✗ Muon masses: MISMATCH")
            print(f"  Original: {orig_vals}")
            print(f"  CCCL:     {cccl_vals}")
    else:
        muon_match = False
        print("✗ Muon masses: MISMATCH (different lengths)")

    print("="*60)
    if electron_match and muon_match:
        print("✓✓✓ SUCCESS: All results match! ✓✓✓")
    else:
        print("✗✗✗ FAILURE: Results do not match! ✗✗✗")
    print("="*60)
