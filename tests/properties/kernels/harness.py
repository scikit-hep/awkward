# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Shared machinery for the kernel property tests.

Each kernel's pure-Python reference implementation is taken straight from
``kernel-specification.yml`` (the single source of truth), so the tests never
depend on the generated ``awkward-cpp/tests-spec/kernels.py``.

The same property runs against every available backend: the compiled CPU
kernel always, and the CUDA kernel when a GPU is present (selected in CI with
``-m cuda``). Both runners look a kernel up by the same key as awkward itself:
the kernel name followed by the dtypes of its array arguments, in order.
"""

from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

SPEC_PATH = Path(__file__).parents[3] / "kernel-specification.yml"


def load_reference(name: str):
    """Exec the kernel's reference ``definition`` from the spec and return it.

    Each reference implementation mutates its output argument(s) in place and
    needs only ``uint8`` in scope (mirroring the prelude that
    ``dev/generate-tests.py`` injects into the generated ``kernels.py``).
    """
    import numpy

    spec = yaml.safe_load(SPEC_PATH.read_text())
    definition = next(k["definition"] for k in spec["kernels"] if k["name"] == name)
    namespace: dict = {"numpy": numpy, "uint8": numpy.uint8}
    exec(definition, namespace)
    return namespace[name]


def _key(name: str, args) -> tuple:
    return (name, *(a.dtype.type for a in args if isinstance(a, np.ndarray)))


def run_cpu(name: str, *args) -> list:
    """Run the compiled CPU kernel on ``args`` (numpy arrays and scalars).

    Return ``args`` with the arrays holding their contents after the call.
    """
    from awkward._backends.numpy import NumpyBackend

    backend = NumpyBackend.instance()
    backend.maybe_kernel_error(backend[_key(name, args)](*args))  # raises on error
    return list(args)


def run_cuda(name: str, *args) -> list:
    """Run the compiled CUDA kernel on copies of ``args`` (numpy arrays and scalars).

    Return ``args`` with the arrays copied back from the device after the call.
    """
    import cupy

    import awkward._connect.cuda as ak_cu
    from awkward._backends.cupy import CupyBackend

    on_device = [cupy.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    CupyBackend.instance()[_key(name, args)](*on_device)
    ak_cu.synchronize_cuda()  # kernel errors surface here
    return [cupy.asnumpy(a) if isinstance(a, cupy.ndarray) else a for a in on_device]


def backends():
    """CPU always; CUDA only when a GPU device is actually present."""
    params = [pytest.param(run_cpu, id="cpu")]
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            params.append(pytest.param(run_cuda, id="cuda", marks=pytest.mark.cuda))
    except Exception:
        pass
    return params
