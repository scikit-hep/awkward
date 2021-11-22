# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_numpy(array, allow_missing=True):
    """
    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a NumPy array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #type), they can be losslessly
    converted to a NumPy array and this function returns without an error.

    Otherwise, the function raises an error. It does not create a NumPy
    array with dtype `"O"` for `np.object_` (see the
    [note on object_ type](https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in))
    since silent conversions to dtype `"O"` arrays would not only be a
    significant performance hit, but would also break functionality, since
    nested lists in a NumPy `"O"` array are severed from the array and
    cannot be sliced as dimensions.

    If `array` is a scalar, it is converted into a NumPy scalar.

    If `allow_missing` is True; NumPy
    [masked arrays](https://docs.scipy.org/doc/numpy/reference/maskedarray.html)
    are a possible result; otherwise, missing values (None) cause this
    function to raise an error.

    See also #ak.from_numpy and #ak.to_cupy.
    """
    layout = ak._v2.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    return layout.to_numpy(allow_missing=allow_missing)
    #
    # if isinstance(array, (bool, str, bytes, numbers.Number)):
    #     return numpy.array([array])[0]
    #
    # elif ak._v2._util.py27 and isinstance(array, ak._v2._util.unicode):
    #     return numpy.array([array])[0]
    #
    # elif isinstance(array, np.ndarray):
    #     return array
    #
    # elif isinstance(array, ak._v2.highlevel.Array):
    #     return to_numpy(array.layout, allow_missing=allow_missing)
    #
    # elif isinstance(array, ak._v2.highlevel.Record):
    #     out = array.layout
    #     return to_numpy(out.array[out.at : out.at + 1], allow_missing=allow_missing)[0]
    #
    # elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
    #     return to_numpy(array.snapshot().layout, allow_missing=allow_missing)
    #
    # elif isinstance(array, ak.layout.ArrayBuilder):
    #     return to_numpy(array.snapshot(), allow_missing=allow_missing)
    #
    # elif ak._v2.operations.describe.parameters(array).get("__array__") == "bytestring":
    #     return numpy.array(
    #         [
    #             ak._v2.behaviors.string.ByteBehavior(array[i]).__bytes__()
    #             for i in range(len(array))
    #         ]
    #     )
    #
    # elif ak._v2.operations.describe.parameters(array).get("__array__") == "string":
    #     return numpy.array(
    #         [
    #             ak._v2.behaviors.string.CharBehavior(array[i]).__str__()
    #             for i in range(len(array))
    #         ]
    #     )
    #
    # elif (
    #     str(ak._v2.operations.describe.type(array)) == "datetime64"
    #     or str(ak._v2.operations.describe.type(array)) == "timedelta64"
    # ):
    #     return array
    #
    # elif isinstance(array, ak.partition.PartitionedArray):   # NO PARTITIONED ARRAY
    #     tocat = [to_numpy(x, allow_missing=allow_missing) for x in array.partitions]
    #     if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
    #         return numpy.ma.concatenate(tocat)
    #     else:
    #         return numpy.concatenate(tocat)
    #
    # elif isinstance(array, ak._v2._util.virtualtypes):
    #     return to_numpy(array.array, allow_missing=True)
    #
    # elif isinstance(array, ak._v2._util.unknowntypes):
    #     return numpy.array([])
    #
    # elif isinstance(array, ak._v2._util.indexedtypes):
    #     return to_numpy(array.project(), allow_missing=allow_missing)
    #
    # elif isinstance(array, ak._v2._util.uniontypes):
    #     contents = [
    #         to_numpy(array.project(i), allow_missing=allow_missing)
    #         for i in range(array.numcontents)
    #     ]
    #
    #     if any(isinstance(x, numpy.ma.MaskedArray) for x in contents):
    #         try:
    #             out = numpy.ma.concatenate(contents)
    #         except Exception:
    #             raise ValueError(
    #                 "cannot convert {0} into numpy.ma.MaskedArray".format(array)
    #
    #             )
    #     else:
    #         try:
    #             out = numpy.concatenate(contents)
    #         except Exception:
    #             raise ValueError(
    #                 "cannot convert {0} into np.ndarray".format(array)
    #
    #             )
    #
    #     tags = numpy.asarray(array.tags)
    #     for tag, content in enumerate(contents):
    #         mask = tags == tag
    #         out[mask] = content
    #     return out
    #
    # elif isinstance(array, ak._v2.contents.UnmaskedArray):
    #     content = to_numpy(array.content, allow_missing=allow_missing)
    #     if allow_missing:
    #         return numpy.ma.MaskedArray(content)
    #     else:
    #         return content
    #
    # elif isinstance(array, ak._v2._util.optiontypes):
    #     content = to_numpy(array.project(), allow_missing=allow_missing)
    #
    #     shape = list(content.shape)
    #     shape[0] = len(array)
    #     data = numpy.empty(shape, dtype=content.dtype)
    #     mask0 = numpy.asarray(array.bytemask()).view(np.bool_)
    #     if mask0.any():
    #         if allow_missing:
    #             mask = numpy.broadcast_to(
    #                 mask0.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape
    #             )
    #             if isinstance(content, numpy.ma.MaskedArray):
    #                 mask1 = numpy.ma.getmaskarray(content)
    #                 mask = mask.copy()
    #                 mask[~mask0] |= mask1
    #
    #             data[~mask0] = content
    #             return numpy.ma.MaskedArray(data, mask)
    #         else:
    #             raise ValueError(
    #                 "ak.to_numpy cannot convert 'None' values to "
    #                 "np.ma.MaskedArray unless the "
    #                 "'allow_missing' parameter is set to True"
    #
    #             )
    #     else:
    #         if allow_missing:
    #             return numpy.ma.MaskedArray(content)
    #         else:
    #             return content
    #
    # elif isinstance(array, ak._v2.contents.RegularArray):
    #     out = to_numpy(array.content, allow_missing=allow_missing)
    #     head, tail = out.shape[0], out.shape[1:]
    #     if array.size == 0:
    #         shape = (0, 0) + tail
    #     else:
    #         shape = (head // array.size, array.size) + tail
    #     return out[: shape[0] * array.size].reshape(shape)
    #
    # elif isinstance(array, ak._v2._util.listtypes):
    #     return to_numpy(array.toRegularArray(), allow_missing=allow_missing)
    #
    # elif isinstance(array, ak._v2._util.recordtypes):
    #     if array.numfields == 0:
    #         return numpy.empty(len(array), dtype=[])
    #     contents = [
    #         to_numpy(array.field(i), allow_missing=allow_missing)
    #         for i in range(array.numfields)
    #     ]
    #     if any(len(x.shape) != 1 for x in contents):
    #         raise ValueError(
    #             "cannot convert {0} into np.ndarray".format(array)
    #
    #         )
    #     out = numpy.empty(
    #         len(contents[0]),
    #         dtype=[(str(n), x.dtype) for n, x in zip(array.keys(), contents)],
    #     )
    #
    #     mask = None
    #     for n, x in zip(array.keys(), contents):
    #         if isinstance(x, numpy.ma.MaskedArray):
    #             if mask is None:
    #                 mask = numpy.ma.zeros(
    #                     len(array), [(n, np.bool_) for n in array.keys()]
    #                 )
    #             if x.mask is not None:
    #                 mask[n] |= x.mask
    #         out[n] = x
    #
    #     if mask is not None:
    #         out = numpy.ma.MaskedArray(out, mask)
    #
    #     return out
    #
    # elif isinstance(array, ak._v2.contents.NumpyArray):
    #     out = ak.nplike.of(array).asarray(array)
    #     if type(out).__module__.startswith("cupy."):
    #         return out.get()
    #     else:
    #         return out
    #
    # elif isinstance(array, ak._v2.contents.Content):
    #     raise AssertionError(
    #         "unrecognized Content type: {0}".format(type(array))
    #
    #     )
    #
    # elif isinstance(array, Iterable):
    #     return numpy.asarray(array)
    #
    # else:
    #     raise ValueError(
    #         "cannot convert {0} into np.ndarray".format(array)
    #
    #     )
