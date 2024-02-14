# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from awkward._backends.backend import Backend
from awkward._backends.dispatch import (
    common_backend,
    regularize_backend,
)
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import TYPE_CHECKING, Any, NamedTuple, Self, TypeVar
from awkward.errors import AxisError

if TYPE_CHECKING:
    from awkward.contents import Content

np = NumpyMetadata.instance()
numpy = Numpy.instance()
numpy_backend = NumpyBackend.instance()


T = TypeVar("T")

if TYPE_CHECKING:
    from awkward.highlevel import Array
    from awkward.highlevel import Record as HighLevelRecord


class HighLevelMetadata(NamedTuple):
    backend: Backend
    attrs: Mapping | None
    behavior: Mapping | None


K = TypeVar("K")
V = TypeVar("V")


def merge_mappings(
    mappings: Sequence[Mapping[K, V]], default: Mapping[K, V] | None = None
) -> Mapping[K, V]:
    # TODO: add zero-copy optimisation
    if len(mappings) == 0:
        return default
    elif len(mappings) == 1:
        return mappings[0]
    else:
        return {k: v for mapping in mappings for k, v in mapping.items()}


class HighLevelContext:
    def __init__(
        self, behavior: Mapping | None = None, attrs: Mapping[str, Any] | None = None
    ):
        self._behavior = behavior
        self._attrs = attrs
        self._is_finalized = False

        self._attrs_from_objects = []
        self._behavior_from_objects = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def _ensure_finalized(self):
        if not self._is_finalized:
            raise RuntimeError("HighLevelContext has not yet been finalized")

    def _ensure_not_finalized(self):
        if self._is_finalized:
            raise RuntimeError("HighLevelContext has already been finalized")

    @property
    def attrs(self) -> Mapping[str, Any] | None:
        self._ensure_finalized()
        return self._attrs

    @property
    def behavior(self) -> Mapping | None:
        self._ensure_finalized()
        return self._behavior

    def finalize(self) -> Self:
        self._ensure_not_finalized()

        if self._behavior is None:
            # TODO: cleaner reverse
            behavior = merge_mappings(self._behavior_from_objects[::-1], default=None)
        else:
            behavior = self._behavior

        if self._attrs is None:
            attrs = merge_mappings(self._attrs_from_objects[::-1], default=None)
        else:
            attrs = self._attrs

        self._attrs = attrs
        self._behavior = behavior
        self._is_finalized = True

        return self

    def update(self, obj: T) -> T:
        from awkward.highlevel import Array, ArrayBuilder, Record

        self._ensure_not_finalized()

        if isinstance(obj, (Array, Record, ArrayBuilder)):
            if obj._attrs is not None:
                self._attrs_from_objects.append(obj._attrs)

            if obj._behavior is not None:
                self._behavior_from_objects.append(obj._behavior)

        return obj

    def unwrap(
        self,
        obj: Any,
        *,
        allow_record: bool = True,
        allow_unknown: bool = False,
        none_policy: Literal["error", "promote", "pass-through"] = "error",
        primitive_policy: Literal["error", "promote", "pass-through"] = "promote",
        string_policy: Literal[
            "error", "promote", "pass-through", "as-characters"
        ] = "as-characters",
        use_from_iter: bool = True,
        regulararray: bool = True,
    ) -> Any:
        from awkward.operations.ak_to_layout import _impl as to_layout_impl

        self.update(obj)

        return to_layout_impl(
            obj,
            allow_record=allow_record,
            allow_unknown=allow_unknown,
            none_policy=none_policy,
            use_from_iter=use_from_iter,
            primitive_policy=primitive_policy,
            string_policy=string_policy,
            regulararray=regulararray,
        )

    def wrap(
        self, obj: Any, *, highlevel: bool = True, allow_other: bool = False
    ) -> Any:
        self._ensure_finalized()

        return wrap_layout(
            obj,
            highlevel=highlevel,
            attrs=self._attrs,
            behavior=self._behavior,
            allow_other=allow_other,
        )


def ensure_same_backend(
    *layouts: Any, default_backend: str | Backend = "cpu"
) -> list[Any]:
    """

    Returns:
        object:
    """
    backends: set[Backend] = {
        layout.backend for layout in layouts if hasattr(layout, "backend")
    }

    backend: Backend
    if len(backends) >= 1:
        backend = common_backend(backends)
    else:
        backend = regularize_backend(default_backend)

    return [
        layout.to_backend(backend) if hasattr(layout, "to_backend") else layout
        for layout in layouts
    ]


def wrap_layout(
    content: T,
    behavior: Mapping | None = None,
    highlevel: bool = True,
    like: Any = None,
    allow_other: bool = False,
    attrs: Mapping | None = None,
) -> T | Array | HighLevelRecord:
    import awkward.highlevel
    from awkward.contents import Content
    from awkward.record import Record

    assert isinstance(content, (Content, Record)) or allow_other
    assert behavior is None or isinstance(behavior, Mapping)
    assert isinstance(highlevel, bool)
    if highlevel:
        if like is not None and behavior is None:
            behavior = behavior_of(like)

        if isinstance(content, Content):
            return awkward.highlevel.Array(content, behavior=behavior, attrs=attrs)
        elif isinstance(content, Record):
            return awkward.highlevel.Record(content, behavior=behavior, attrs=attrs)
        elif allow_other:
            return content
        else:
            raise AssertionError

    elif isinstance(content, (Content, Record)) or allow_other:
        return content
    else:
        raise AssertionError


def maybe_highlevel_to_lowlevel(obj):
    """
    Args:
        obj: an object

    Calls #ak.to_layout and returns the result iff. the object is a high-level
    Awkward object, otherwise the object is returned as-is.

    This function should be removed once scalars are properly handled by `to_layout`.
    """
    import awkward.highlevel

    if isinstance(
        obj,
        (
            awkward.highlevel.Array,
            awkward.highlevel.Record,
            awkward.highlevel.ArrayBuilder,
        ),
    ):
        return awkward.to_layout(obj)
    else:
        return obj


def from_arraylib(array, regulararray, recordarray):
    from awkward.contents import (
        ByteMaskedArray,
        ListArray,
        NumpyArray,
        RecordArray,
        RegularArray,
        UnmaskedArray,
    )
    from awkward.index import Index8, Index64

    # overshadows global NumPy import for nplike-safety
    nplike = nplike_of_obj(array)

    def recurse(array, mask=None):
        if regulararray and len(array.shape) > 1:
            new_shape = (-1,) + array.shape[2:]
            return RegularArray(
                recurse(nplike.reshape(array, new_shape), mask),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            array = nplike.reshape(array, (1,))

        if array.dtype.kind == "S":
            assert nplike is numpy
            asbytes = array.reshape(-1)
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = numpy.add(starts, numpy.char.str_len(asbytes))
            data = ListArray(
                Index64(starts),
                Index64(stops),
                NumpyArray(
                    asbytes.view("u1"),
                    parameters={"__array__": "byte"},
                    backend=numpy_backend,
                ),
                parameters={"__array__": "bytestring"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = RegularArray(data, array.shape[i], array.shape[i - 1])

        elif array.dtype.kind == "U":
            assert nplike is numpy
            asbytes = numpy.char.encode(array.reshape(-1), "utf-8", "surrogateescape")
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = numpy.add(starts, numpy.char.str_len(asbytes))
            data = ListArray(
                Index64(starts),
                Index64(stops),
                NumpyArray(
                    asbytes.view("u1"),
                    parameters={"__array__": "char"},
                    backend=numpy_backend,
                ),
                parameters={"__array__": "string"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = RegularArray(data, array.shape[i], array.shape[i - 1])

        else:
            data = NumpyArray(array)

        if mask is None:
            return data

        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return UnmaskedArray(data)
            else:

                def attach(x):
                    if isinstance(x, NumpyArray):
                        return UnmaskedArray(x)
                    else:
                        return RegularArray(attach(x.content), x.size, len(x))

                return attach(data.to_RegularArray())

        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return ByteMaskedArray(Index8(mask), data, valid_when=False)

    if array.dtype == np.dtype("O"):
        raise TypeError("Awkward Array does not support arrays with object dtypes.")

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        array = numpy.ma.getdata(array)
        if isinstance(mask, np.ndarray) and len(mask.shape) > 1:
            regulararray = True
            mask = mask.reshape(-1)
    else:
        mask = None

    if not recordarray or array.dtype.names is None:
        layout = recurse(array, mask)

    else:
        contents = []
        for name in array.dtype.names:
            contents.append(recurse(array[name], mask))
        layout = RecordArray(contents, array.dtype.names)

    return layout


def maybe_posaxis(layout: Content, axis: int, depth: int) -> int | None:
    from awkward.record import Record

    if isinstance(layout, Record):
        if axis == 0:
            raise AxisError("Record type at axis=0 is a scalar, not an array")
        return maybe_posaxis(layout._array, axis, depth)

    if axis >= 0:
        return axis

    else:
        is_branching, additional_depth = layout.branch_depth
        if not is_branching:
            return axis + depth + additional_depth - 1
        else:
            return None
