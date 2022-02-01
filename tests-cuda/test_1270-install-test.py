# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import ctypes


def test_import_and_loading_shared_lib():
    try:
        import awkward_cuda_kernels
    except ModuleNotFoundError:
        pytest.fail("Could not import awkward_cuda_kernels")

    try:
        import ctypes

        ctypes.cdll.LoadLibrary(awkward_cuda_kernels.shared_library_path)
    except Exception:
        pytest.fail("Could not load the shared library")
