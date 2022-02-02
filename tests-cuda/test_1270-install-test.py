# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401


def test_import_and_loading_shared_lib():
    try:
        import awkward_cuda_kernels
    except ModuleNotFoundError:
        pytest.fail("Could not import awkward_cuda_kernels")

    try:
        import ctypes

        cuak = ctypes.cdll.LoadLibrary(awkward_cuda_kernels.shared_library_path)

        if not hasattr(cuak, "awkward_cuda_ptr_device_name"):
            pytest.fail("Cannot access the functions in the loaded shared library")
    except Exception:
        pytest.fail("Could not load the shared library or access the function")
