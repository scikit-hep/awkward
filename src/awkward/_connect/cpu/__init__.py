# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU (NumPy) backend for the lazy IR.

Mirrors ``awkward._connect.cuda``: ``lazy()`` wraps a cpu-backed array in a
``LazyAwkwardArray``, and ``ir_nodes``/``helpers`` provide the NumPy fast
paths corresponding to the CCCL ones.
"""

from __future__ import annotations


def lazy(array):
    """Wrap a CPU-backed array for lazy, fused execution.

    Internal entry point (``awkward._connect.cpu.lazy``); not exposed as a public
    ``ak.*`` name while the lazy/fusion work is a PoC.

    Args:
        array (ak.Array): A CPU-backed array.

    Returns a :class:`awkward._connect.lazy._lazy_impl.LazyAwkwardArray`.

    Raises:
        TypeError: If ``array`` is not on the CPU backend.
    """
    if array.layout.backend.name != "cpu":
        raise TypeError("cpu.lazy is only available for arrays with the CPU backend")

    from awkward._connect.lazy._lazy_impl import lazy as _lazy

    return _lazy(array)


__all__ = ("lazy",)
