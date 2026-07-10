# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CUDA-specific lazy IR nodes.

The generic execution machinery (``IRNode``, ``compute``, ``reset_cache``,
``walk``, ``topological_order``, ``Input``) lives in
``awkward._connect.lazy.core``; the nodes here lower to the cuda.compute
(CCCL) primitives in ``awkward._connect.cuda.helpers`` — one implementation,
shared with the eager helper functions.
"""

from __future__ import annotations

from awkward._connect.lazy.core import (  # noqa: F401
    Input,
    IRNode,
    compute,
    reset_cache,
    topological_order,
    walk,
)

from .helpers import (
    awkward_to_cccl_iterator,
    empty_like,
    filter_lists,
    list_sizes,
    select_lists,
    transform_lists,
)


class EmptyLike(IRNode):
    """Lazy node that allocates an empty array shaped like its input."""

    __slots__ = ()

    def lower(self, array):
        return empty_like(array)


class ToIterator(IRNode):
    """Lazy node that builds a cuda.compute iterator over its input."""

    __slots__ = ()

    def lower(self, array):
        return awkward_to_cccl_iterator(array)


class Filter(IRNode):
    """Keep elements within each list for which ``predicate`` is true.

    Args:
        array: The input node/array to filter.
        predicate (callable): A callable compiled with numba.cuda for the CCCL
            select path (not a materialized boolean array).
    """

    __slots__ = ("predicate",)

    def __init__(self, array, predicate):
        super().__init__(array)
        self.predicate = predicate

    def lower(self, array):
        return filter_lists(array, self.predicate)


class SelectLists(IRNode):
    """Keep entire lists selected by a per-list mask.

    Args:
        array: The input node/array whose lists are selected.
        mask: A per-list mask node/array.
    """

    __slots__ = ()

    def __init__(self, array, mask):
        super().__init__(array, mask)

    def lower(self, array, mask):
        return select_lists(array, mask)


class ListSizes(IRNode):
    """Lazy node that yields per-list element counts."""

    __slots__ = ()

    def lower(self, array):
        return list_sizes(array)


class TransformLists(IRNode):
    """Apply an n-ary op across the items of equal-size lists.

    Args:
        array: The input node/array of equal-size lists.
        out: Pre-allocated output buffer (one value per list).
        list_size (int): The common list length.
        op (callable): Device op taking one argument per item position.
    """

    __slots__ = ("op",)

    def __init__(self, array, out, list_size, op):
        super().__init__(array, out, list_size)
        self.op = op

    def lower(self, array, out, list_size):
        return transform_lists(array, out, list_size, self.op)
