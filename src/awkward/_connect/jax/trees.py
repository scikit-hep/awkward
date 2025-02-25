# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import jax

import awkward as ak
from awkward import contents, highlevel, record
from awkward._backends.jax import JaxBackend
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Generic, TypeVar, Union
from awkward.contents import Content

T = TypeVar(
    "T", bound=Union[highlevel.Array, highlevel.Record, contents.Content, record.Record]
)

np = NumpyMetadata.instance()


def split_buffers(buffers: dict) -> tuple[dict, dict]:
    data_buffers, other_buffers = {}, {}
    for key, buf in buffers.items():
        _, attr = key.rsplit("-", 1)
        if attr == "data":
            data_buffers[key] = buf
        else:
            other_buffers[key] = buf
    return data_buffers, other_buffers


class AuxData(Generic[T]):
    """
    This class is used to store the auxiliary data needed to reconstruct an Awkward Array from its buffers.

    AuxData deliberately can not use the layout, because the layout has a reference to the buffers which are replaced
    by JAX with tracers - this leads to tracer leaks!

    Instead, this class holds the form, length, and other numpy buffers needed to reconstruct it.
    """

    def __init__(
        self,
        data_buffer_keys: tuple[str],
        other_buffers: dict,
        form: ak.forms.Form,
        length: int,
        ctx: HighLevelContext,
        highlevel: bool = True,
    ):
        self._data_buffer_keys = data_buffer_keys
        self._other_buffers = other_buffers
        self._form = form
        self._length = length
        self._ctx = ctx
        self._highlevel = highlevel

    @classmethod
    def from_array_or_layout(cls, obj: T):
        with HighLevelContext() as ctx:
            layout = ctx.unwrap(obj, allow_record=True, primitive_policy="error")

        # change backend
        jax_backend = JaxBackend.instance()
        layout = layout.to_backend(jax_backend)

        # decompose
        form, length, buffers = ak.operations.to_buffers(layout)

        # we need to split buffers into all "data" buffers and all others (e.g. index, offsets, etc.)
        data_buffers, other_buffers = split_buffers(buffers)

        # now we need to flatten the data buffers
        data_buffer_keys, data_flat_buffers = zip(*data_buffers.items())
        return data_flat_buffers, AuxData(
            data_buffer_keys=data_buffer_keys,
            other_buffers=other_buffers,
            form=form,
            length=length,
            ctx=ctx,
            highlevel=not isinstance(obj, Content),
        )

    def unflatten(self, data_buffers: tuple) -> T:
        for buffer in data_buffers:
            # Check that JAX isn't trying to give us float0 types
            dtype = getattr(buffer, "dtype", None)
            if dtype == np.dtype([("float0", "V")]):
                raise TypeError(
                    f"a buffer with the dtype {buffer.dtype} was encountered during unflattening. "
                    "JAX uses this dtype for the tangents of integer/boolean outputs; these cannot "
                    "reasonably be differentiated. Make sure that you are not computing the derivative "
                    "of a boolean/integer (array) valued function."
                )

        # reconstitute data buffers
        data_buffers = dict(zip(self._data_buffer_keys, data_buffers))

        # combine data buffers with other buffers
        buffers = {**self._other_buffers, **data_buffers}

        # reconstruct layout
        layout = ak.operations.from_buffers(
            self._form,
            self._length,
            buffers,
            backend="jax",
            highlevel=False,  # we will wrap it later
        )

        # wrap layout
        return self._ctx.wrap(
            layout,
            highlevel=self._highlevel,
            allow_other=True,
        )

    def __eq__(self, other: AuxData) -> bool:
        return (
            self._form.is_equal_to(
                other=other._form, all_parameters=True, form_key=True
            )
            and self._length == other._length
        )

    def __ne__(self, other: AuxData) -> bool:
        return not self == other


def jax_flatten(
    array: T,
) -> tuple[tuple, AuxData]:
    result = AuxData.from_array_or_layout(array)
    return result


def jax_unflatten(aux_data: AuxData, children: tuple) -> T | None:
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
