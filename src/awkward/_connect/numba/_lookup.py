# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ctypes

import numpy as np

from awkward._nplikes.numpy import Numpy


# The buffer protocol supplies an actual pointer, including for read-only arrays.
# Keep buf pointer-typed: c_void_p fields otherwise become Python integer addresses.
class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.PYFUNCTYPE(
    ctypes.c_int, ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int
)(("PyObject_GetBuffer", ctypes.pythonapi))
_release_buffer = ctypes.PYFUNCTYPE(None, ctypes.POINTER(_PyBuffer))(
    ("PyBuffer_Release", ctypes.pythonapi)
)


def make_buffer_table(positions, nplike):
    """Build an owned pointer table, indexed in parallel with layout metadata.

    NumPy exposes the table's storage for transport/unboxing only; lowering loads
    its entries as pointers, never integers. ``positions`` keeps every buffer alive.
    CUDA's host interface supplies device addresses as integers; copy these into
    device pointer storage before launching a kernel, never reconstructing them
    from integers inside generated code.
    """
    if nplike is not Numpy.instance():
        addresses = np.asarray(
            [0 if isinstance(x, int) else nplike.memory_ptr(x) for x in positions],
            dtype=np.intp,
        )
        return nplike.asarray(addresses.view(np.uint8))

    table = (ctypes.POINTER(ctypes.c_uint8) * len(positions))()
    for i, array in enumerate(positions):
        if not isinstance(array, int):
            buffer = _PyBuffer()
            _get_buffer(array, ctypes.byref(buffer), 0)
            try:
                table[i] = buffer.buf
            finally:
                _release_buffer(ctypes.byref(buffer))
    return np.frombuffer(table, dtype=np.uint8)
