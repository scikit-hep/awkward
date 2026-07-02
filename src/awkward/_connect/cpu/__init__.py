# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU (NumPy) backend for the lazy IR.

Mirrors ``awkward._connect.cuda``: ``lazy()`` wraps a cpu-backed array in a
``LazyAwkwardArray``, and ``ir_nodes``/``helpers`` provide the NumPy fast
paths corresponding to the CCCL ones.
"""

from __future__ import annotations


def lazy(array):
    if array.layout.backend.name != "cpu":
        raise TypeError("ak.cpu.lazy is only available for arrays with the CPU backend")

    from awkward._connect.lazy._lazy_impl import lazy as _lazy

    return _lazy(array)


__all__ = ("lazy",)
