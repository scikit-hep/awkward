# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "src/awkward/_connect/cuda"
GENERATOR = ROOT / "dev/generate-kernel-signatures.py"
KERNEL = "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64"


def test_raw_kernel_registry_matches_remaining_sources():
    tree = ast.parse(GENERATOR.read_text())
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "cuda_kernels_impl" for t in n.targets
        )
    )
    registered = ast.literal_eval(assignment.value)
    sources = {p.stem for p in (CUDA / "cuda_kernels").glob("awkward_*.cu")}
    assert len(registered) == len(set(registered))
    assert set(registered) == sources


@pytest.mark.thread_unsafe
def test_nonlocal_ranges_generate_compute_dispatch(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    generator = runpy.run_path(str(GENERATOR))
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
