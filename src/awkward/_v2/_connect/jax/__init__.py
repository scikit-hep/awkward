# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

try:
    import jax

    error_message = None

except ModuleNotFoundError:
    jax = None
    error_message = """to use {0}, you must install jax:

    pip install jax jaxlib

or

    conda install -c conda-forge jax jaxlib
"""

pytrees_registered = False


def register_pytrees():
    jax.tree_util.register_pytree_node(
        ak._v2.contents.bitmaskedarray.BitMaskedArray,
        ak._v2.contents.bitmaskedarray.BitMaskedArray.jax_flatten,
        ak._v2.contents.bitmaskedarray.BitMaskedArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.bytemaskedarray.ByteMaskedArray,
        ak._v2.contents.bytemaskedarray.ByteMaskedArray.jax_flatten,
        ak._v2.contents.bytemaskedarray.ByteMaskedArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.emptyarray.EmptyArray,
        ak._v2.contents.emptyarray.EmptyArray.jax_flatten,
        ak._v2.contents.emptyarray.EmptyArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.indexedarray.IndexedArray,
        ak._v2.contents.indexedarray.IndexedArray.jax_flatten,
        ak._v2.contents.indexedarray.IndexedArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.indexedoptionarray.IndexedOptionArray,
        ak._v2.contents.indexedoptionarray.IndexedOptionArray.jax_flatten,
        ak._v2.contents.indexedoptionarray.IndexedOptionArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.numpyarray.NumpyArray,
        ak._v2.contents.numpyarray.NumpyArray.jax_flatten,
        ak._v2.contents.numpyarray.NumpyArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.listarray.ListArray,
        ak._v2.contents.listarray.ListArray.jax_flatten,
        ak._v2.contents.listarray.ListArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.listoffsetarray.ListOffsetArray,
        ak._v2.contents.listoffsetarray.ListOffsetArray.jax_flatten,
        ak._v2.contents.listoffsetarray.ListOffsetArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.recordarray.RecordArray,
        ak._v2.contents.recordarray.RecordArray.jax_flatten,
        ak._v2.contents.recordarray.RecordArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.regulararray.RegularArray,
        ak._v2.contents.regulararray.RegularArray.jax_flatten,
        ak._v2.contents.regulararray.RegularArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.unionarray.UnionArray,
        ak._v2.contents.unionarray.UnionArray.jax_flatten,
        ak._v2.contents.unionarray.UnionArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.contents.unmaskedarray.UnmaskedArray,
        ak._v2.contents.unmaskedarray.UnmaskedArray.jax_flatten,
        ak._v2.contents.unmaskedarray.UnmaskedArray.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.highlevel.Array,
        ak._v2.highlevel.Array.jax_flatten,
        ak._v2.highlevel.Array.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.record.Record,
        ak._v2.record.Record.jax_flatten,
        ak._v2.record.Record.jax_unflatten,
    )
    jax.tree_util.register_pytree_node(
        ak._v2.highlevel.Record,
        ak._v2.highlevel.Record.jax_flatten,
        ak._v2.highlevel.Record.jax_unflatten,
    )


def import_jax(name="Awkward Arrays with JAX"):
    if jax is None:
        raise ImportError(error_message.format(name))

    global pytrees_registered

    if not pytrees_registered:
        register_pytrees()
        pytrees_registered = True
    return jax


def _find_numpyarray_nodes(layout):

    data_ptrs = []

    def find_nparray_ptrs(node, **kwargs):
        if isinstance(node, ak._v2.contents.numpyarray.NumpyArray):
            data_ptrs.append(node.data)

    layout.recursively_apply(action=find_nparray_ptrs, return_array=False)

    return data_ptrs


def _replace_numpyarray_nodes(layout, buffers):
    def replace_numpyarray_nodes(node, **kwargs):
        if isinstance(node, ak._v2.contents.numpyarray.NumpyArray):
            buffer = buffers[0]
            buffers.pop(0)
            return ak._v2.contents.NumpyArray(
                buffer,
                layout.identifier,
                layout.parameters,
                nplike=ak.nplike.Jax.instance(),
            )

    return layout.recursively_apply(action=replace_numpyarray_nodes)


class AuxData:
    def __init__(self, layout):
        self._layout = layout

    @property
    def layout(self):
        return self._layout

    def __eq__(self, other):
        def is_layout_eq(self_layout, other_layout):
            if self_layout.__class__ is not other_layout.__class__:
                return False
            elif isinstance(self_layout, ak._v2.contents.BitMaskedArray):
                return is_layout_eq(
                    self_layout.mask, other_layout.mask
                ) and is_layout_eq(self_layout.content, other_layout.content)
            elif isinstance(self_layout, ak._v2.contents.ByteMaskedArray):
                return is_layout_eq(
                    self_layout.mask, other_layout.mask
                ) and is_layout_eq(self_layout.content, other_layout.content)
            elif isinstance(self_layout, ak._v2.contents.EmptyArray):
                return True
            elif isinstance(self_layout, ak._v2.contents.IndexedArray):
                return is_layout_eq(
                    self_layout.index, other_layout.index
                ) and is_layout_eq(self_layout.content, other_layout.content)
            elif isinstance(self_layout, ak._v2.contents.IndexedOptionArray):
                return (
                    is_layout_eq(self_layout.mask, other_layout.mask)
                    and is_layout_eq(self_layout.index, other_layout.index)
                    and is_layout_eq(self_layout.content, other_layout.content)
                )
            elif isinstance(self_layout, ak._v2.contents.ListArray):
                return (
                    is_layout_eq(self_layout.starts, other_layout.starts)
                    and is_layout_eq(self_layout.stops, other_layout.stops)
                    and is_layout_eq(self_layout.content, other_layout.content)
                )
            elif isinstance(self_layout, ak._v2.contents.ListOffsetArray):
                return is_layout_eq(
                    self_layout.offsets, other_layout.offsets
                ) and is_layout_eq(self_layout.content, other_layout.content)
            elif isinstance(self_layout, ak._v2.contents.RecordArray):
                return self_layout.fields == other_layout.fields and all(
                    [
                        is_layout_eq(self_layout.contents[i], other_layout.contents[i])
                        for i in range(len(self_layout.contents))
                    ]
                )
            elif isinstance(self_layout, ak._v2.contents.UnionArray):
                return (
                    is_layout_eq(self_layout.tags, other_layout.tags)
                    and is_layout_eq(self_layout.index, other_layout.index)
                    and all(
                        [
                            is_layout_eq(
                                self_layout.contents[i], other_layout.contents[i]
                            )
                            for i in range(len(self_layout.contents))
                        ]
                    )
                )
            elif isinstance(self_layout, ak._v2.contents.NumpyArray):
                return len(self_layout) == len(other_layout)
            elif isinstance(self_layout, ak._v2.index.Index):
                return self_layout.nplike.array_equal(
                    self_layout.data, other_layout.data
                )
            else:
                raise ak._v2._util.error(AssertionError("Content Type not recognized."))

        return is_layout_eq(self.layout, other.layout)
