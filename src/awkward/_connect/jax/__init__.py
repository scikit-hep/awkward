# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def find_numpyarray_nodes(layout):

    data_ptrs = []

    def find_nparray_ptrs(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
            data_ptrs.append(node.data)

    layout.recursively_apply(action=find_nparray_ptrs, return_array=False)

    return data_ptrs


def replace_numpyarray_nodes(layout, buffers):
    def replace_numpyarray_nodes(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
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
