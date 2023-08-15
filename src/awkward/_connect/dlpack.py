# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("DLPackDevice", "get_layout_device", "to_dlpack")
from enum import IntEnum

from awkward._do import recursively_apply
from awkward._typing import Any
from awkward.contents import Content


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


def get_layout_device(layout: Content) -> tuple[int, int]:
    device: tuple[int, int] | None = None

    def apply(layout, **kwargs):
        nonlocal device
        if layout.is_numpy:
            device = layout.data.__dlpack_device__()

    recursively_apply(
        layout,
        apply,
        return_simplified=False,
        numpy_to_regular=False,
        return_array=False,
    )

    if device is None:
        raise TypeError(
            "Cannot determine the DLPack device for an array "
            "without any NumpyArray layout nodes"
        )
    return device


def to_dlpack(layout: Content, stream: Any) -> Any:
    array = layout.to_backend_array(allow_missing=False)
    return array.__dlpack__(stream)
