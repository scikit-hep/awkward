# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ast
import pathlib

import pytest

yaml = pytest.importorskip("yaml")

# The cuda.compute host implementations in _connect/cuda/_compute.py are
# called positionally with exactly the spec-defined kernel arguments
# (CudaComputeKernel does `self._impl(*args)`). A kernel-specification.yml
# signature change that is not propagated to the wrapper fails only at GPU
# runtime ("missing 1 required positional argument"), which CPU-only CI never
# sees. This test catches the mismatch by parsing source, with no CUDA (or
# cuda.compute) required.

REPO = pathlib.Path(__file__).parent.parent
SPEC = REPO / "kernel-specification.yml"
CUPY_BACKEND = REPO / "src" / "awkward" / "_backends" / "cupy.py"
COMPUTE = REPO / "src" / "awkward" / "_connect" / "cuda" / "_compute.py"


def _spec_signatures():
    """kernel/specialization name -> list of spec argument names."""
    with open(SPEC) as f:
        spec = yaml.safe_load(f)["kernels"]
    out = {}
    for kernel in spec:
        first = kernel["specializations"][0]
        args = [a["name"] for a in first["args"]]
        out[kernel["name"]] = args
        for specialization in kernel["specializations"]:
            out[specialization["name"]] = [a["name"] for a in specialization["args"]]
    return out


def _dispatch_table():
    """kernel name -> _compute.py function name, from _get_cuda_compute_impl."""
    tree = ast.parse(CUPY_BACKEND.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_cuda_compute_impl":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    table = {}
                    for key, value in zip(sub.keys, sub.values, strict=True):
                        if isinstance(key, ast.Constant) and isinstance(
                            value, ast.Attribute
                        ):
                            table[key.value] = value.attr
                    return table
    raise AssertionError("_get_cuda_compute_impl dispatch dict not found")


def _compute_signatures():
    """_compute.py function name -> list of parameter names."""
    tree = ast.parse(COMPUTE.read_text())
    return {
        node.name: [arg.arg for arg in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def test_cuda_compute_wrappers_match_kernel_spec():
    spec = _spec_signatures()
    table = _dispatch_table()
    impls = _compute_signatures()

    assert table, "empty dispatch table"
    problems = []
    for kernel_name, impl_name in table.items():
        if kernel_name not in spec:
            problems.append(f"{kernel_name}: not in kernel-specification.yml")
            continue
        if impl_name not in impls:
            problems.append(f"{kernel_name}: {impl_name} not found in _compute.py")
            continue
        n_spec, n_impl = len(spec[kernel_name]), len(impls[impl_name])
        if n_spec != n_impl:
            problems.append(
                f"{kernel_name}: spec has {n_spec} args {spec[kernel_name]} but "
                f"_compute.{impl_name} takes {n_impl} {impls[impl_name]}"
            )
    assert not problems, "\n".join(problems)
