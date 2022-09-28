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
        ak.contents.bitmaskedarray.BitMaskedArray,
        ak.contents.bytemaskedarray.ByteMaskedArray,
        ak.contents.emptyarray.EmptyArray,
        ak.contents.indexedarray.IndexedArray,
        ak.contents.indexedoptionarray.IndexedOptionArray,
        ak.contents.numpyarray.NumpyArray,
        ak.contents.listarray.ListArray,
        ak.contents.listoffsetarray.ListOffsetArray,
        ak.contents.recordarray.RecordArray,
        ak.contents.unionarray.UnionArray,
        ak.contents.unmaskedarray.UnmaskedArray,
        ak.record.Record,
    ]:
        jax.tree_util.register_pytree_node(
            cls,
            cls.jax_flatten,
            cls.jax_unflatten,
        )

    for cls in [ak.highlevel.Array, ak.highlevel.Record]:
        jax.tree_util.register_pytree_node(
            cls,
            cls._jax_flatten,
            cls._jax_unflatten,
        )


def import_jax(name="Awkward Arrays with JAX"):
    if jax is None:
        raise ak._util.error(ModuleNotFoundError(error_message.format(name)))

    global pytrees_registered

    if not pytrees_registered:
        register_pytrees()
        pytrees_registered = True
    return jax


def _find_numpyarray_nodes(layout):

    data_ptrs = []

    def find_nparray_ptrs(node, **kwargs):
        if isinstance(node, ak.contents.numpyarray.NumpyArray):
            data_ptrs.append(node.data)

    layout.recursively_apply(action=find_nparray_ptrs, return_array=False)

    return data_ptrs


def _replace_numpyarray_nodes(layout, buffers):
    def replace_numpyarray_nodes(node, **kwargs):
        if isinstance(node, ak.contents.numpyarray.NumpyArray):
            buffer = buffers[0]
            buffers.pop(0)
            return ak.contents.NumpyArray(
                buffer,
                layout.identifier,
                layout.parameters,
                nplike=ak.nplikes.Jax.instance(),
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
