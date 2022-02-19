# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: no change; keep this file.

import ctypes
import platform

import awkward._kernel_signatures_cuda

if platform.system() == "Windows":
    raise ValueError(
        "There is no support for Awkward CUDA on Windows. Please use a Linux system to use CUDA "
        "accelerated Awkward Arrays."
    )
elif platform.system() == "Darwin":
    raise ValueError(
        "There is no support for Awkward CUDA on MacOS. Please use a Linux system to use CUDA "
        "accelerated Awkward Arrays."
    )
else:
    try:
        import awkward_cuda_kernels
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            """to use CUDA awkward arrays in Python, install the 'awkward[cuda]' package with:
    pip install awkward[cuda] --upgrade
or
    conda install -c conda-forge awkward-cuda-kernels"""
        ) from None

lib = ctypes.cdll.LoadLibrary(awkward_cuda_kernels.shared_library_path)
kernel = awkward._kernel_signatures_cuda.by_signature(lib)
