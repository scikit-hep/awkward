# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import jax

import awkward as ak
from awkward import _errors, _nplikes, contents, highlevel, record
from awkward.typing import Generic, TypeVar, Union

numpy = _nplikes.Numpy.instance()
np = _nplikes.NumpyMetadata.instance()


def find_all_buffers(
    layout: contents.Content | record.Record,
) -> list[numpy.ndarray]:

    data_ptrs = []

    def action(node, **kwargs):
        if isinstance(node, ak.contents.NumpyArray):
            data_ptrs.append(node.data)

    ak._do.recursively_apply(
        layout, action=action, return_array=False, numpy_to_regular=False
    )

    return data_ptrs


def replace_all_buffers(
    layout: contents.Content | record.Record,
    buffers: list,
    backend: ak._backends.Backend,
):
    def action(node, **kwargs):
        jaxlike = _nplikes.Jax.instance()
        if isinstance(node, ak.contents.NumpyArray):
            buffer = buffers.pop(0)
            # JAX might give us non-buffers, so ignore them
            if not (numpy.is_own_array(buffer) or jaxlike.is_own_array(buffer)):
                return
            else:
                return ak.contents.NumpyArray(
                    buffer, parameters=node.parameters, backend=backend
                )

    return ak._do.recursively_apply(layout, action=action, numpy_to_regular=False)


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
        is_highlevel = isinstance(obj, (highlevel.Array, highlevel.Record))
        if is_highlevel:
            layout = obj.layout
        elif isinstance(obj, (contents.Content, record.Record)):
            layout = obj
        else:
            raise _errors.wrap_error(TypeError)

        # First, make sure we're all JAX
        jax_backend = ak._backends.JaxBackend.instance()
        layout = layout.to_backend(jax_backend)

        # Now pull out the Jax tracers / arrays
        buffers = find_all_buffers(layout)

        # # Drop the references to the existing buffers by replacing them with empty buffers
        # # FIXME: This works-around the fact that AuxData should probably contain only a form and length,
        # # rather than the actual layout (which holds references to the buffers that we're returning)
        # # We use NumPy buffers here to ensure that we don't create any new tracers (they're just placeholders)
        # # This is particularly unpleasant, because we're mixing nplikes here (deliberately)
        # # We should use `to_buffers`.
        # import numpy as _numpy
        #
        # def create_placeholder_like(array) -> _numpy.ndarray:
        #     data = _numpy.empty(1, dtype=array.dtype)
        #     strides = tuple(0 for _ in array.shape)
        #     return _numpy.lib.stride_tricks.as_strided(
        #         data, array.shape, strides=strides, writeable=False
        #     )
        # layout = replace_all_buffers(
        #     layout,
        #     [create_placeholder_like(n) for n in buffers],
        #     nplike=_nplikes.Numpy.instance(),
        # )

        return buffers, AuxData(
            layout=layout,
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

    def unflatten(self, buffers: tuple) -> T:
        for buffer in buffers:
            # Check that JAX isn't trying to give us float0 types
            dtype = getattr(buffer, "dtype", None)
            if dtype == np.dtype([("float0", "V")]):
                raise _errors.wrap_error(
                    TypeError(
                        f"a buffer with the dtype {buffer.dtype} was encountered during unflattening. "
                        "JAX uses this dtype for the tangents of integer/boolean outputs; these cannot "
                        "reasonably be differentiated. Make sure that you are not computing the derivative "
                        "of a boolean/integer (array) valued function."
                    )
                )

        # Replace the mixed NumPy-JAX layout leaves with the given buffers (and use the JAX nplike)
        layout = replace_all_buffers(
            self._layout, list(buffers), backend=ak._backends.JaxBackend.instance()
        )
        return ak._util.wrap(
            layout, behavior=self._behavior, highlevel=self._is_highlevel
        )

    def __eq__(self, other: AuxData) -> bool:
        return self.layout.is_equal_to(
            other.layout, index_dtype=False, numpyarray=False
        )


def jax_flatten(
    array: T,
) -> tuple[list[numpy.ndarray], AuxData]:
    result = AuxData.from_array_or_layout(array)
    return result


def jax_unflatten(aux_data: AuxData, children: list[numpy.ndarray]) -> T | None:
    return aux_data.unflatten(children)


HighLevelType = TypeVar(
    "HighLevelType", bound="type[highlevel.Array | highlevel.Record]"
)


def register_pytree_class(cls: T) -> T:
    """
    Args:
        cls: class to register with JAX

    Return the class, after registering it with JAX.

    """
    jax.tree_util.register_pytree_node(
        cls,
        jax_flatten,
        jax_unflatten,
    )
    return cls
