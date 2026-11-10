# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import runpy
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "src/awkward/_connect/cuda"
GENERATOR = ROOT / "dev/generate-kernel-signatures.py"
KERNEL = "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64"

# Kernels in the specification that have neither a raw CUDA source nor a
# cuda.compute implementation. Shrink this set when one gains CUDA support.
KNOWN_UNSUPPORTED_ON_CUDA = {
    "awkward_IndexedArray_unique_next_index_and_offsets_64",
    "awkward_ListOffsetArray_argsort_strings",
    "awkward_ListOffsetArray_getitem_boolmask",
    "awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",
    "awkward_NumpyArray_sort_asstrings_uint8",
    "awkward_NumpyArray_unique_strings",
    "awkward_unique_offsets",
    "awkward_unique_ranges",
    "awkward_unique_ranges_bool",
}


@pytest.fixture(scope="module")
def generator():
    pytest.importorskip("yaml")
    return runpy.run_path(str(GENERATOR))


def test_raw_kernel_registry_matches_remaining_sources(generator):
    registered = generator["cuda_kernels_impl"]
    sources = {p.stem for p in (CUDA / "cuda_kernels").glob("awkward_*.cu")}
    assert len(registered) == len(set(registered))
    assert set(registered) == sources


@pytest.mark.thread_unsafe
def test_kernels_without_raw_source_dispatch_to_cuda_compute(generator, monkeypatch):
    yaml = pytest.importorskip("yaml")
    import awkward._connect.cuda
    from awkward._backends.cupy import CupyBackend

    # Stand in for cuda.compute so the dispatch table can be built without a GPU.
    class StubCompute(types.ModuleType):
        def __getattr__(self, name):
            return name

    stub = StubCompute("awkward._connect.cuda._compute")
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(awkward._connect.cuda, "_compute", stub, raising=False)

    specification = yaml.safe_load((ROOT / "kernel-specification.yml").read_text())
    names = {kernel["name"] for kernel in specification["kernels"]}
    raw = set(generator["cuda_kernels_impl"])
    backend = object.__new__(CupyBackend)

    uncovered = {
        name
        for name in names - raw
        if not backend._supports_cuda_compute(name)
        or backend._get_cuda_compute_impl(name) is None
    }
    assert uncovered == KNOWN_UNSUPPORTED_ON_CUDA


@pytest.mark.thread_unsafe
def test_nonlocal_ranges_generate_compute_dispatch(tmp_path, monkeypatch, generator):
    yaml = pytest.importorskip("yaml")
    generate = generator["kernel_signatures_cuda_py"]
    monkeypatch.setitem(generate.__globals__, "CURRENT_DIR", str(tmp_path / "dev"))
    destination = tmp_path / "src/awkward/_connect/cuda"
    shutil.copytree(CUDA / "cuda_kernels", destination / "cuda_kernels")
    specification = yaml.safe_load((ROOT / "kernel-specification.yml").read_text())
    specification["kernels"] = [
        kernel for kernel in specification["kernels"] if kernel["name"] == KERNEL
    ]
    assert len(specification["kernels"]) == 1
    generate(specification)
    tree = ast.parse((destination / "_kernel_signatures.py").read_text())
    registrations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "out"
            for target in node.targets
        )
    ]
    assert len(registrations) == len(specification["kernels"][0]["specializations"])
    assert all(
        isinstance(n.value, ast.Constant) and n.value.value is None
        for n in registrations
    )
