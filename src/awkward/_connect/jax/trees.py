# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from typing import Generic, TypeVar, Union

import awkward as ak
from awkward import _errors, contents, highlevel, nplikes, record

numpy = nplikes.Numpy.instance()


def find_all_buffers(
    layout: contents.Content | record.Record,
) -> list[numpy.ndarray]:

    data_ptrs = []

    def action(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
            data_ptrs.append(node.data)

    layout.recursively_apply(action=action, return_array=False)

    return data_ptrs


def replace_all_buffers(
    layout: contents.Content | record.Record, buffers: list[numpy.ndarray]
):
    nplike = nplikes.nplike_of(*buffers)

    def action(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
            return ak.contents.NumpyArray(
                buffers.pop(0), layout.identifier, layout.parameters, nplike=nplike
            )

    return layout.recursively_apply(action=action)


T = TypeVar(
    "T", bound=Union[highlevel.Array, highlevel.Record, contents.Content, record.Record]
)


class AuxData(Generic[T]):
    def __init__(
        self,
        layout: contents.Content | record.Record,
        is_highlevel: bool,
        behavior: dict | None = None,
    ):
        self._layout = layout
        self._behavior = behavior
        self._is_highlevel = is_highlevel

    @classmethod
    def from_array_or_layout(cls, obj: T):
        import numpy

        is_highlevel = isinstance(obj, (highlevel.Array, highlevel.Record))
        if is_highlevel:
            layout = obj.layout
        elif isinstance(obj, (contents.Content, record.Record)):
            layout = obj
        else:
            raise _errors.wrap_error(TypeError)

        buffers = find_all_buffers(layout)
        # Drop the references to the existing buffers by replacing them with empty buffers
        # FIXME: This works-around the fact that AuxData should probably contain only a form and length,
        # rather than the actual layout (which holds references to the buffers that we're returning)
        # We use NumPy buffers here to ensure that we don't create any new tracers (they're just placeholders)
        # We should use `to_buffers`.

        def create_placeholder_like(array) -> numpy.ndarray:
            data = numpy.empty(1, dtype=array.dtype)
            strides = tuple(0 for _ in array.shape)
            return numpy.lib.stride_tricks.as_strided(
                data, array.shape, strides=strides, writeable=False
            )

        return buffers, AuxData(
            layout=replace_all_buffers(
                layout, [create_placeholder_like(n) for n in buffers]
            ),
            is_highlevel=is_highlevel,
            behavior=ak._util.behavior_of(obj),
        )

    @property
    def layout(self) -> contents.Content | record.Record:
        return self._layout

    @property
    def behavior(self) -> dict | None:
        return self._behavior

    @property
    def is_highlevel(self) -> bool:
        return self._is_highlevel

    def unflatten(self, buffers) -> T:
        layout = replace_all_buffers(self._layout, list(buffers))
        return ak._util.wrap(
            layout, behavior=self._behavior, highlevel=self._is_highlevel
        )

    def __eq__(self, other: AuxData) -> bool:
        return self.layout.layout_equal(
            other.layout, index_dtype=False, numpyarray=False
        )


def jax_flatten(
    array: T,
) -> tuple[list[numpy.ndarray], AuxData]:
    result = AuxData.from_array_or_layout(array)
    return result


def jax_unflatten(aux_data: AuxData, children: list[numpy.ndarray]) -> T:
    return aux_data.unflatten(children)
