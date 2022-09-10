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
    for cls in [
        ak._v2.contents.bitmaskedarray.BitMaskedArray,
        ak._v2.contents.bytemaskedarray.ByteMaskedArray,
        ak._v2.contents.emptyarray.EmptyArray,
        ak._v2.contents.indexedarray.IndexedArray,
        ak._v2.contents.indexedoptionarray.IndexedOptionArray,
        ak._v2.contents.numpyarray.NumpyArray,
        ak._v2.contents.listarray.ListArray,
        ak._v2.contents.listoffsetarray.ListOffsetArray,
        ak._v2.contents.recordarray.RecordArray,
        ak._v2.contents.unionarray.UnionArray,
        ak._v2.contents.unmaskedarray.UnmaskedArray,
        ak._v2.record.Record,
    ]:
        jax.tree_util.register_pytree_node(
            cls,
            cls.jax_flatten,
            cls.jax_unflatten,
        )

    for cls in [ak._v2.highlevel.Array, ak._v2.highlevel.Record]:
        jax.tree_util.register_pytree_node(
            cls,
            cls._jax_flatten,
            cls._jax_unflatten,
        )


def import_jax(name="Awkward Arrays with JAX"):
    if jax is None:
        raise ak._v2._util.error(ModuleNotFoundError(error_message.format(name)))

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
        return self.layout.layout_equal(
            other.layout, index_dtype=False, numpyarray=False
        )
