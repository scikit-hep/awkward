# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("DLPackDevice",)
from enum import IntEnum


class DLPackDevice(IntEnum):
    CPU = 1  # CPU
    CUDA = 2  # GPU
    CUDA_PINNED = 3  # GPU & CPU
    OPENCL = 4  # UNSUPPORTED
    VULKAN = 7  # UNSUPPORTED
    METAL = 8  # UNSUPPORTED
    VPI = 9  # UNSUPPORTED
    ROCM = 10  # GPU
    ROCM_PINNED = 11  # GPU & CPU
    CUDA_MANAGED = 13  # GPU & CPU
