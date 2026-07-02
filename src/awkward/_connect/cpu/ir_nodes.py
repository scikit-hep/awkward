# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU-specific lazy IR nodes.

The generic execution machinery (``IRNode``, ``compute``, ``reset_cache``,
``walk``, ``topological_order``, ``Input``) lives in
``awkward._connect.lazy.core``; the nodes here lower to the NumPy
implementations in ``awkward._connect.cpu.helpers``.  The node set mirrors
``awkward._connect.cuda.ir_nodes``.
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
    empty_like,
    filter_lists,
    list_sizes,
    select_lists,
    transform_lists,
)


class EmptyLike(IRNode):
    __slots__ = ()

    def lower(self, array):
        return empty_like(array)


class Filter(IRNode):
    """Keep elements within each list for which `predicate` is true.

    `predicate` must be vectorized over the flat content
    (e.g. ``lambda x: x > 2``).
    """

    __slots__ = ("predicate",)

    def __init__(self, array, predicate):
        super().__init__(array)
        self.predicate = predicate

    def lower(self, array):
        return filter_lists(array, self.predicate)


class SelectLists(IRNode):
    """Keep entire lists selected by a per-list mask."""

    __slots__ = ()

    def __init__(self, array, mask):
        super().__init__(array, mask)

    def lower(self, array, mask):
        return select_lists(array, mask)


class ListSizes(IRNode):
    __slots__ = ()

    def lower(self, array):
        return list_sizes(array)


class TransformLists(IRNode):
    """Apply an n-ary op across the items of equal-size lists."""

    __slots__ = ("op",)

    def __init__(self, array, out, list_size, op):
        super().__init__(array, out, list_size)
        self.op = op

    def lower(self, array, out, list_size):
        return transform_lists(array, out, list_size, self.op)
