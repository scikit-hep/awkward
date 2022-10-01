# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward import contents, highlevel, nplikes, record

numpy = nplikes.Numpy.instance()


def find_numpyarray_nodes(
    layout: contents.Content | record.Record,
) -> list[numpy.ndarray]:

    data_ptrs = []

    def find_nparray_ptrs(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
            data_ptrs.append(node.data)

    layout.recursively_apply(action=find_nparray_ptrs, return_array=False)

    return data_ptrs


def replace_numpyarray_nodes(
    layout: contents.Content | record.Record, buffers: list[numpy.ndarray]
):
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
    def __init__(self, layout: contents.Content | record.Record, behavior=None):
        self._layout = layout

    @property
    def layout(self) -> contents.Content | record.Record:
        return self._layout

    def __eq__(self, other: AuxData) -> bool:
        return self.layout.layout_equal(
            other.layout, index_dtype=False, numpyarray=False
        )


def jax_flatten_highlevel(
    array: highlevel.Array | highlevel.Record,
) -> tuple[list[numpy.ndarray], AuxData]:
    return array._layout._jax_flatten()


def jax_unflatten_highlevel(
    aux_data: AuxData, children: list[numpy.ndarray]
) -> highlevel.Array | highlevel.Record:
    return ak._util.wrap(aux_data.layout.jax_unflatten(aux_data, children))
