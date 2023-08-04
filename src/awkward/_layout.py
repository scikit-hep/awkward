# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections.abc import Mapping

from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._nplikes.dispatch import nplike_of
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.errors import AxisError

np = NumpyMetadata.instance()
numpy = Numpy.instance()
numpy_backend = NumpyBackend.instance()


def wrap_layout(content, behavior=None, highlevel=True, like=None, allow_other=False):
    import awkward.highlevel
    from awkward.contents import Content
    from awkward.record import Record

    assert content is None or isinstance(content, (Content, Record)) or allow_other
    assert behavior is None or isinstance(behavior, Mapping)
    assert isinstance(highlevel, bool)
    if highlevel:
        if like is not None and behavior is None:
            behavior = behavior_of(like)

        if isinstance(content, Content):
            return awkward.highlevel.Array(content, behavior=behavior)
        elif isinstance(content, Record):
            return awkward.highlevel.Record(content, behavior=behavior)

    return content


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
    nplike = nplike_of(array)

    def recurse(array, mask=None):
        cls = type(array)
        if Jax.is_tracer_type(cls):
            raise TypeError("Jax tracers cannot be used with `ak.from_arraylib`")

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


def maybe_posaxis(layout, axis, depth):
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
