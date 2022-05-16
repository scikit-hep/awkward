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
        ak._v2.Array,
        ak._v2.Array.jax_flatten,
        ak._v2.Array.jax_unflatten,
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
            node._data = buffers[0]
            buffers.pop()

    return layout.recursively_apply(action=replace_numpyarray_nodes)


class AuxData:
    layout = None

    def __init__(self, layout):
        self.layout = layout

    def __eq__(self, other):
        return (
            len(self.layout) == len(other.layout)
            and self.layout.form == other.layout.form
        )
