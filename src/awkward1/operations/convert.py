# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers
import json
import collections
import math

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1.layout
import awkward1._ext
import awkward1._util


def from_numpy(array, regulararray=False, highlevel=True, behavior=None):
    """
    Args:
        array (np.ndarray): The NumPy array to convert into an Awkward Array.
            This array can be a np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Converts a NumPy array into an Awkward Array.

    The resulting layout may involve the following #ak.layout.Content types
    (only):

       * #ak.layout.NumpyArray
       * #ak.layout.ByteMaskedArray or #ak.layout.UnmaskedArray if the
         `array` is an np.ma.MaskedArray.
       * #ak.layout.RegularArray if `regulararray=True`.

    See also #ak.to_numpy.
    """

    def recurse(array, mask):
        if regulararray and len(array.shape) > 1:
            return awkward1.layout.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:]), mask), array.shape[1]
            )

        if len(array.shape) == 0:
            data = awkward1.layout.NumpyArray(array.reshape(1))
        else:
            data = awkward1.layout.NumpyArray(array)

        if mask is None:
            return data
        elif mask is False:
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            return awkward1.layout.UnmaskedArray(data)
        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return awkward1.layout.ByteMaskedArray(
                awkward1.layout.Index8(mask), data, valid_when=False
            )

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        array = numpy.ma.getdata(array)
        if isinstance(mask, numpy.ndarray) and len(mask.shape) > 1:
            regulararray = True
            mask = mask.reshape(-1)
    else:
        mask = None

    layout = recurse(array, mask)
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout


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

    See also #ak.from_numpy.
    """
    import awkward1.highlevel

    if isinstance(array, (bool, str, bytes, numbers.Number)):
        return numpy.array([array])[0]

    elif awkward1._util.py27 and isinstance(array, awkward1._util.unicode):
        return numpy.array([array])[0]

    elif isinstance(array, numpy.ndarray):
        return array

    elif isinstance(array, awkward1.highlevel.Array):
        return to_numpy(array.layout, allow_missing=allow_missing)

    elif isinstance(array, awkward1.highlevel.Record):
        out = array.layout
        return to_numpy(out.array[out.at : out.at + 1], allow_missing=allow_missing)[0]

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return to_numpy(array.snapshot().layout, allow_missing=allow_missing)

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return to_numpy(array.snapshot(), allow_missing=allow_missing)

    elif (
        awkward1.operations.describe.parameters(array).get("__array__") == "bytestring"
    ):
        return numpy.array(
            [
                awkward1.behaviors.string.ByteBehavior(array[i]).__bytes__()
                for i in range(len(array))
            ]
        )

    elif awkward1.operations.describe.parameters(array).get("__array__") == "string":
        return numpy.array(
            [
                awkward1.behaviors.string.CharBehavior(array[i]).__str__()
                for i in range(len(array))
            ]
        )

    elif isinstance(array, awkward1.partition.PartitionedArray):
        tocat = [to_numpy(x, allow_missing=allow_missing) for x in array.partitions]
        if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
            return numpy.ma.concatenate(tocat)
        else:
            return numpy.concatenate(tocat)

    elif isinstance(array, awkward1._util.virtualtypes):
        return to_numpy(array.array, allow_missing=True)

    elif isinstance(array, awkward1._util.unknowntypes):
        return numpy.array([])

    elif isinstance(array, awkward1._util.indexedtypes):
        return to_numpy(array.project(), allow_missing=allow_missing)

    elif isinstance(array, awkward1._util.uniontypes):
        contents = [
            to_numpy(array.project(i), allow_missing=allow_missing)
            for i in range(array.numcontents)
        ]

        if any(isinstance(x, numpy.ma.MaskedArray) for x in contents):
            try:
                out = numpy.ma.concatenate(contents)
            except Exception:
                raise ValueError(
                    "cannot convert {0} into numpy.ma.MaskedArray".format(array)
                )
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception:
                raise ValueError("cannot convert {0} into numpy.ndarray".format(array))

        tags = numpy.asarray(array.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    elif isinstance(array, awkward1.layout.UnmaskedArray):
        content = to_numpy(array.content, allow_missing=allow_missing)
        if allow_missing:
            return numpy.ma.MaskedArray(content)
        else:
            return content

    elif isinstance(array, awkward1._util.optiontypes):
        content = to_numpy(array.project(), allow_missing=allow_missing)
        shape = list(content.shape)
        shape[0] = len(array)
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = numpy.asarray(array.bytemask()).view(numpy.bool_)
        if mask0.any():
            if allow_missing:
                mask = numpy.broadcast_to(
                    mask0.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape
                )
                data[~mask0] = content
                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ValueError(
                    "to_numpy cannot convert 'None' values to "
                    "np.ma.MaskedArray unless the "
                    "'allow_missing' parameter is set to True"
                )
        else:
            if allow_missing:
                return numpy.ma.MaskedArray(content)
            else:
                return content

    elif isinstance(array, awkward1.layout.RegularArray):
        out = to_numpy(array.content, allow_missing=allow_missing)
        head, tail = out.shape[0], out.shape[1:]
        shape = (head // array.size, array.size) + tail
        return out[: shape[0] * array.size].reshape(shape)

    elif isinstance(array, awkward1._util.listtypes):
        return to_numpy(array.toRegularArray(), allow_missing=allow_missing)

    elif isinstance(array, awkward1._util.recordtypes):
        if array.numfields == 0:
            return numpy.empty(len(array), dtype=[])
        contents = [
            to_numpy(array.field(i), allow_missing=allow_missing)
            for i in range(array.numfields)
        ]
        if any(len(x.shape) != 1 for x in contents):
            raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
        out = numpy.empty(
            len(contents[0]),
            dtype=[(str(n), x.dtype) for n, x in zip(array.keys(), contents)],
        )
        for n, x in zip(array.keys(), contents):
            out[n] = x
        return out

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array)

    elif isinstance(array, awkward1.layout.Content):
        raise AssertionError("unrecognized Content type: {0}".format(type(array)))

    elif isinstance(array, Iterable):
        return numpy.asarray(array)

    else:
        raise ValueError("cannot convert {0} into numpy.ndarray".format(array))


def from_iter(
    iterable, highlevel=True, behavior=None, allow_record=True, initial=1024, resize=1.5
):
    """
    Args:
        iterable (Python iterable): Data to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.
        allow_record (bool): If True, the outermost element may be a record
            (returning #ak.Record or #ak.layout.Record type, depending on
            `highlevel`); if False, the outermost element must be an array.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.

    Converts Python data into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested Python data can be converted, but the output will never
    have regular-typed array lengths.

    The following Python types are supported.

       * bool, including `np.bool_`: converted into #ak.layout.NumpyArray.
       * int, including `np.integer`: converted into #ak.layout.NumpyArray.
       * float, including `np.floating`: converted into #ak.layout.NumpyArray.
       * bytes: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"bytestring"` (unencoded bytes).
       * str: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"string"` (UTF-8 encoded string).
       * tuple: converted into #ak.layout.RecordArray without field names
         (i.e. homogeneously typed, uniform sized tuples).
       * dict: converted into #ak.layout.RecordArray with field names
         (i.e. homogeneously typed records with the same sets of fields).
       * iterable, including np.ndarray: converted into
         #ak.layout.ListOffsetArray.

    See also #ak.to_list.
    """
    if isinstance(iterable, dict):
        if allow_record:
            return from_iter(
                [iterable],
                highlevel=highlevel,
                behavior=behavior,
                initial=initial,
                resize=resize,
            )[0]
        else:
            raise ValueError("cannot produce an array from a dict")
    out = awkward1.layout.ArrayBuilder(initial=initial, resize=resize)
    for x in iterable:
        out.fromiter(x)
    layout = out.snapshot()
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout


def to_list(array):
    """
    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into Python objects.

    Awkward Array types have the following Pythonic translations.

       * #ak.types.PrimitiveType: converted into bool, int, float.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into list.
       * #ak.types.RegularType: also converted into list. Python (and JSON)
         forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"`: converted into bytes.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__string__"`: converted into str.
       * #ak.types.RecordArray without field names: converted into tuple.
       * #ak.types.RecordArray with field names: converted into dict.
       * #ak.types.UnionArray: Python data are naturally heterogeneous.

    See also #ak.from_iter and #ak.Array.tolist.
    """
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif awkward1._util.py27 and isinstance(array, awkward1._util.unicode):
        return array

    elif isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.behaviors.string.ByteBehavior):
        return array.__bytes__()

    elif isinstance(array, awkward1.behaviors.string.CharBehavior):
        return array.__str__()

    elif awkward1.operations.describe.parameters(array).get("__array__") == "byte":
        return awkward1.behaviors.string.CharBehavior(array).__bytes__()

    elif awkward1.operations.describe.parameters(array).get("__array__") == "char":
        return awkward1.behaviors.string.CharBehavior(array).__str__()

    elif isinstance(array, awkward1.highlevel.Array):
        return [to_list(x) for x in array]

    elif isinstance(array, awkward1.highlevel.Record):
        return to_list(array.layout)

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return to_list(array.snapshot())

    elif isinstance(array, awkward1.layout.Record) and array.istuple:
        return tuple(to_list(x) for x in array.fields())

    elif isinstance(array, awkward1.layout.Record):
        return {n: to_list(x) for n, x in array.fielditems()}

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return [to_list(x) for x in array.snapshot()]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(
        array, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
    ):
        return [to_list(x) for x in array]

    elif isinstance(array, dict):
        return dict((n, to_list(x)) for n, x in array.items())

    elif isinstance(array, Iterable):
        return [to_list(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(type(array)))


def from_json(
    source, highlevel=True, behavior=None, initial=1024, resize=1.5, buffersize=65536
):
    """
    Args:
        source (str): JSON-formatted string to convert into an array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts a JSON string into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested JSON can be converted, but the output will never have
    regular-typed array lengths.

    See also #ak.to_json.
    """
    layout = awkward1._ext.fromjson(
        source, initial=initial, resize=resize, buffersize=buffersize
    )
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout


def to_json(array, destination=None, pretty=False, maxdecimals=None, buffersize=65536):
    """
    Args:
        array: Data to convert to JSON.
        destination (None or str): If None, this function returns a JSON str;
            if a str, it uses that as a file name and writes (overwrites) that
            file (returning None).
        pretty (bool): If True, indent the output for human readability; if
            False, output compact JSON without spaces.
        maxdecimals (None or int): If an int, limit the number of
            floating-point decimals to this number; if None, write all digits.
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a JSON string or file.

    Awkward Array types have the following JSON translations.

       * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into JSON lists.
       * #ak.types.RegularType: also converted into JSON lists. JSON (and
         Python) forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"` or `"__string__"`: converted into JSON strings.
       * #ak.types.RecordArray without field names: converted into JSON
         objects with numbers as strings for keys.
       * #ak.types.RecordArray with field names: converted into JSON objects.
       * #ak.types.UnionArray: JSON data are naturally heterogeneous.

    See also #ak.from_json and #ak.Array.tojson.
    """
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, bytes):
        return json.dumps(array.decode("utf-8", "surrogateescape"))

    elif awkward1._util.py27 and isinstance(array, awkward1._util.unicode):
        return json.dumps(array)

    elif isinstance(array, numpy.ndarray):
        out = awkward1.layout.NumpyArray(array)

    elif isinstance(array, awkward1.highlevel.Array):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.Record):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, awkward1.layout.Record):
        out = array

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        out = array.snapshot()

    elif isinstance(
        array, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
    ):
        out = array

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

    if destination is None:
        return out.tojson(pretty=pretty, maxdecimals=maxdecimals)
    else:
        return out.tojson(
            destination, pretty=pretty, maxdecimals=maxdecimals, buffersize=buffersize
        )


def from_awkward0(
    array, keeplayout=False, regulararray=False, highlevel=True, behavior=None
):
    """
    Args:
        array (Awkward 0.x or Awkward 1.x array): Data to convert to Awkward
            1.x.
        keeplayout (bool): If True, stay true to the Awkward 0.x layout,
            ensuring zero-copy; otherwise, allow transformations that copy
            data for more flexibility.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Converts an array from Awkward 0.x to Awkward 1.x.

    This is only needed during the transition from the old library to the
    new library.

    If `array` is already an Awkward 1.x Array, it is simply passed through
    this function (so that interfaces that scripts don't need to remove this
    function when their 0.x sources are replaced by 1.x).
    """
    # See https://github.com/scikit-hep/awkward-0.x/blob/405b7eaeea51b60947a79c782b1abf0d72f6729b/specification.adoc
    import awkward as awkward0

    # If a source of Awkward0 arrays ever starts emitting Awkward1 arrays
    # (e.g. Uproot), this function turns into a pass-through.
    if isinstance(array, (awkward1.highlevel.Array, awkward1.highlevel.Record)):
        if highlevel:
            return array
        else:
            return array.layout
    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        if highlevel:
            return array.snapshot()
        else:
            return array._layout.snapshot()
    elif isinstance(array, (awkward1.layout.Content, awkward1.layout.Record)):
        if highlevel:
            return awkward1._util.wrap(array, behavior)
        else:
            return array
    elif isinstance(array, awkward1.layout.ArrayBuilder):
        if highlevel:
            return awkward1._util.wrap(array.snapshot(), behavior)
        else:
            return array.snapshot()

    def recurse(array, level):
        if isinstance(array, dict):
            keys = []
            values = []
            for n, x in array.items():
                keys.append(n)
                if isinstance(
                    x,
                    (
                        dict,
                        tuple,
                        numpy.ma.MaskedArray,
                        numpy.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[numpy.newaxis])
                else:
                    values.append(awkward1.layout.NumpyArray(numpy.array([x])))
            return awkward1.layout.RecordArray(values, keys)[0]

        elif isinstance(array, tuple):
            values = []
            for x in array:
                if isinstance(
                    x,
                    (
                        dict,
                        tuple,
                        numpy.ma.MaskedArray,
                        numpy.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[numpy.newaxis])
                else:
                    values.append(awkward1.layout.NumpyArray(numpy.array([x])))
            return awkward1.layout.RecordArray(values)[0]

        elif isinstance(array, numpy.ma.MaskedArray):
            return from_numpy(array, regulararray=regulararray, highlevel=False)

        elif isinstance(array, numpy.ndarray):
            return from_numpy(array, regulararray=regulararray, highlevel=False)

        elif isinstance(array, awkward0.JaggedArray):
            # starts, stops, content
            # offsetsaliased(starts, stops)
            startsmax = numpy.iinfo(array.starts.dtype.type).max
            stopsmax = numpy.iinfo(array.stops.dtype.type).max
            if (
                len(array.starts.shape) == 1
                and len(array.stops.shape) == 1
                and awkward0.JaggedArray.offsetsaliased(array.starts, array.stops)
            ):
                if startsmax >= from_awkward0.int64max:
                    offsets = awkward1.layout.Index64(array.offsets)
                    return awkward1.layout.ListOffsetArray64(
                        offsets, recurse(array.content, level + 1)
                    )
                elif startsmax >= from_awkward0.uint32max:
                    offsets = awkward1.layout.IndexU32(array.offsets)
                    return awkward1.layout.ListOffsetArrayU32(
                        offsets, recurse(array.content, level + 1)
                    )
                else:
                    offsets = awkward1.layout.Index32(array.offsets)
                    return awkward1.layout.ListOffsetArray32(
                        offsets, recurse(array.content, level + 1)
                    )

            else:
                if (
                    startsmax >= from_awkward0.int64max
                    or stopsmax >= from_awkward0.int64max
                ):
                    starts = awkward1.layout.Index64(array.starts.reshape(-1))
                    stops = awkward1.layout.Index64(array.stops.reshape(-1))
                    out = awkward1.layout.ListArray64(
                        starts, stops, recurse(array.content, level + 1)
                    )
                elif (
                    startsmax >= from_awkward0.uint32max
                    or stopsmax >= from_awkward0.uint32max
                ):
                    starts = awkward1.layout.IndexU32(array.starts.reshape(-1))
                    stops = awkward1.layout.IndexU32(array.stops.reshape(-1))
                    out = awkward1.layout.ListArrayU32(
                        starts, stops, recurse(array.content, level + 1)
                    )
                else:
                    starts = awkward1.layout.Index32(array.starts.reshape(-1))
                    stops = awkward1.layout.Index32(array.stops.reshape(-1))
                    out = awkward1.layout.ListArray32(
                        starts, stops, recurse(array.content, level + 1)
                    )
                for size in array.starts.shape[:0:-1]:
                    out = awkward1.layout.RegularArray(out, size)
                return out

        elif isinstance(array, awkward0.Table):
            # contents
            if array.istuple:
                return awkward1.layout.RecordArray(
                    [recurse(x, level + 1) for x in array.contents.values()]
                )
            else:
                keys = []
                values = []
                for n, x in array.contents.items():
                    keys.append(n)
                    values.append(recurse(x, level + 1))
                return awkward1.layout.RecordArray(values, keys)

        elif isinstance(array, awkward0.UnionArray):
            # tags, index, contents
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_64(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )
            elif indexmax >= from_awkward0.uint32max:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_U32(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )
            else:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_32(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.MaskedArray):
            # mask, content, maskedwhen
            mask = awkward1.layout.Index8(array.mask.view(numpy.int8).reshape(-1))
            out = awkward1.layout.ByteMaskedArray(
                mask,
                recurse(array.content, level + 1),
                valid_when=(not array.maskedwhen),
            )
            for size in array.mask.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.BitMaskedArray):
            # mask, content, maskedwhen, lsborder
            mask = awkward1.layout.IndexU8(array.mask.view(numpy.uint8))
            return awkward1.layout.BitMaskedArray(
                mask,
                recurse(array.content, level + 1),
                valid_when=(not array.maskedwhen),
                length=len(array.content),
                lsb_order=array.lsborder,
            )

        elif isinstance(array, awkward0.IndexedMaskedArray):
            # mask, content, maskedwhen
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArray64(
                    index, recurse(array.content, level + 1)
                )
            elif indexmax >= from_awkward0.uint32max:
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArrayU32(
                    index, recurse(array.content, level + 1)
                )
            else:
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArray32(
                    index, recurse(array.content, level + 1)
                )

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.IndexedArray):
            # index, content
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.IndexedArray64(
                    index, recurse(array.content, level + 1)
                )
            elif indexmax >= from_awkward0.uint32max:
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.IndexedArrayU32(
                    index, recurse(array.content, level + 1)
                )
            else:
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.IndexedArray32(
                    index, recurse(array.content, level + 1)
                )

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.SparseArray):
            # length, index, content, default
            if keeplayout:
                raise ValueError(
                    "awkward1.SparseArray hasn't been written (if at all); "
                    "try keeplayout=False"
                )
            return recurse(array.dense, level + 1)

        elif isinstance(array, awkward0.StringArray):
            # starts, stops, content, encoding
            out = recurse(array._content, level + 1)
            if array.encoding is None:
                out.content.setparameter("__array__", "byte")
                out.setparameter("__array__", "bytestring")
            elif array.encoding == "utf-8":
                out.content.setparameter("__array__", "char")
                out.setparameter("__array__", "string")
            else:
                raise ValueError(
                    "unsupported encoding: {0}".format(repr(array.encoding))
                )
            return out

        elif isinstance(array, awkward0.ObjectArray):
            # content, generator, args, kwargs
            if keeplayout:
                raise ValueError(
                    "there isn't (and won't ever be) an awkward1 equivalent "
                    "of awkward0.ObjectArray; try keeplayout=False"
                )
            out = recurse(array.content, level + 1)
            out.setparameter(
                "__record__",
                getattr(
                    array.generator,
                    "__qualname__",
                    getattr(array.generator, "__name__", repr(array.generator)),
                ),
            )
            return out

        if isinstance(array, awkward0.ChunkedArray):
            # chunks, chunksizes
            if keeplayout and level != 0:
                raise ValueError(
                    "awkward1 PartitionedArrays are only allowed "
                    "at the root of a data structure, unlike "
                    "awkward0.ChunkedArray; try keeplayout=False"
                )
            elif level == 0:
                return awkward1.partition.IrregularlyPartitionedArray(
                    [recurse(x, level + 1) for x in array.chunks]
                )
            else:
                return awkward1.operations.structure.concatenate(
                    [recurse(x, level + 1) for x in array.chunks], highlevel=False
                )

        elif isinstance(array, awkward0.AppendableArray):
            # chunkshape, dtype, chunks
            raise ValueError(
                "the awkward1 equivalent of awkward0.AppendableArray is "
                "awkward1.ArrayBuilder, but it is not a Content type, not "
                "mixable with immutable array elements"
            )

        elif isinstance(array, awkward0.VirtualArray):
            # generator, args, kwargs, cache, persistentkey, type, nbytes, persistvirtual
            if keeplayout:
                raise NotImplementedError("FIXME")
            else:
                return recurse(array.array, level + 1)

        else:
            raise TypeError("not an awkward0 array: {0}".format(repr(array)))

    out = recurse(array, 0)
    if highlevel:
        return awkward1._util.wrap(out, behavior)
    else:
        return out


from_awkward0.int8max = numpy.iinfo(numpy.int8).max
from_awkward0.int32max = numpy.iinfo(numpy.int32).max
from_awkward0.uint32max = numpy.iinfo(numpy.uint32).max
from_awkward0.int64max = numpy.iinfo(numpy.int64).max


def to_awkward0(array, keeplayout=False):
    """
    Args:
        array: Data to convert into an Awkward 0.x array.
        keeplayout (bool): If True, stay true to the Awkward 1.x layout,
            ensuring zero-copy; otherwise, allow transformations that copy
            data for more flexibility.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into an Awkward 0.x array.

    This is only needed during the transition from the old library to the new
    library.
    """
    # See https://github.com/scikit-hep/awkward-0.x/blob/405b7eaeea51b60947a79c782b1abf0d72f6729b/specification.adoc
    import awkward as awkward0

    def recurse(layout):
        if isinstance(layout, awkward1.partition.PartitionedArray):
            return awkward0.ChunkedArray([recurse(x) for x in layout.partitions])

        elif isinstance(layout, awkward1.layout.NumpyArray):
            return numpy.asarray(layout)

        elif isinstance(layout, awkward1.layout.EmptyArray):
            return numpy.array([])

        elif isinstance(layout, awkward1.layout.RegularArray):
            # content, size
            if keeplayout:
                raise ValueError(
                    "awkward0 has no equivalent of RegularArray; "
                    "try keeplayout=False"
                )
            offsets = numpy.arange(0, (len(layout) + 1) * layout.size, layout.size)
            return awkward0.JaggedArray.fromoffsets(offsets, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListArray32):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, awkward1.layout.ListArrayU32):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, awkward1.layout.ListArray64):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, awkward1.layout.ListOffsetArray32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.ListOffsetArrayU32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.ListOffsetArray64):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.Record):
            # istuple, numfields, field(i)
            out = []
            for i in range(layout.numfields):
                content = layout.field(i)
                if isinstance(
                    content, (awkward1.layout.Content, awkward1.layout.Record)
                ):
                    out.append(recurse(content))
                else:
                    out.append(content)
            if layout.istuple:
                return tuple(out)
            else:
                return dict(zip(layout.keys(), out))

        elif isinstance(layout, awkward1.layout.RecordArray):
            # istuple, numfields, field(i)
            if layout.numfields == 0 and len(layout) != 0:
                raise ValueError(
                    "cannot convert zero-field, nonzero-length RecordArray "
                    "to awkward0.Table (limitation in awkward0)"
                )
            keys = layout.keys()
            values = [recurse(x) for x in layout.contents]
            pairs = collections.OrderedDict(zip(keys, values))
            out = awkward0.Table(pairs)
            if layout.istuple:
                out._rowname = "tuple"
            return out

        elif isinstance(layout, awkward1.layout.UnionArray8_32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, awkward1.layout.UnionArray8_U32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, awkward1.layout.UnionArray8_64):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, awkward1.layout.IndexedOptionArray32):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = index < -1
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedOptionArray64):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = index < -1
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedArray32):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.IndexedArrayU32):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.IndexedArray64):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, awkward1.layout.ByteMaskedArray):
            # mask, content, valid_when
            return awkward0.MaskedArray(
                numpy.asarray(layout.mask),
                recurse(layout.content),
                maskedwhen=(not layout.valid_when),
            )

        elif isinstance(layout, awkward1.layout.BitMaskedArray):
            # mask, content, valid_when, length, lsb_order
            return awkward0.BitMaskedArray(
                numpy.asarray(layout.mask),
                recurse(layout.content),
                maskedwhen=(not layout.valid_when),
                lsborder=layout.lsb_order,
            )

        elif isinstance(layout, awkward1.layout.UnmaskedArray):
            # content
            return recurse(layout.content)  # no equivalent in awkward0

        elif isinstance(layout, awkward1.layout.VirtualArray):
            raise NotImplementedError("FIXME")

        else:
            raise AssertionError(
                "missing converter for {0}".format(type(layout).__name__)
            )

    layout = to_layout(
        array, allow_record=True, allow_other=False, numpytype=(numpy.generic,)
    )
    return recurse(layout)


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(numpy.number, numpy.bool_, numpy.bool),
):
    """
    Args:
        array: Data to convert into an #ak.layout.Content and maybe
            #ak.layout.Record and other types.
        allow_record (bool): If True, allow #ak.layout.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        numpytype (tuple of NumPy types): Allowed NumPy types in
            #ak.layout.NumpyArray outputs.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.layout.Content and maybe #ak.layout.Record and other
    types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis.
    """
    import awkward1.highlevel

    if isinstance(array, awkward1.highlevel.Array):
        return array.layout

    elif allow_record and isinstance(array, awkward1.highlevel.Record):
        return array.layout

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.snapshot()

    elif isinstance(
        array, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
    ):
        return array

    elif allow_record and isinstance(array, awkward1.layout.Record):
        return array

    elif isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        data = numpy.ma.getdata(array)
        if mask is False:
            out = awkward1.layout.UnmaskedArray(
                awkward1.layout.NumpyArray(data.reshape(-1))
            )
        else:
            out = awkward1.layout.ByteMaskedArray(
                awkward1.layout.Index8(mask.reshape(-1)),
                awkward1.layout.NumpyArray(data.reshape(-1)),
            )
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif isinstance(array, numpy.ndarray):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError("NumPy {0} not allowed".format(repr(array.dtype)))
        out = awkward1.layout.NumpyArray(array.reshape(-1))
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif isinstance(array, (str, bytes)) or (
        awkward1._util.py27 and isinstance(array, awkward1._util.unicode)
    ):
        return from_iter([array], highlevel=False)

    elif isinstance(array, Iterable):
        return from_iter(array, highlevel=False)

    elif not allow_other:
        raise TypeError("{0} cannot be converted into an Awkward Array".format(array))

    else:
        return array


def regularize_numpyarray(array, allow_empty=True, highlevel=True):
    """
    Args:
        array: Data to convert into an Awkward Array.
        allow_empty (bool): If True, allow #ak.layout.EmptyArray in the output;
            otherwise, convert empty arrays into #ak.layout.NumpyArray.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Converts any multidimensional #ak.layout.NumpyArray.shape into nested
    #ak.layout.RegularArray nodes. The output may have any Awkward data type:
    this only changes the representation of #ak.layout.NumpyArray.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis.
    """

    def getfunction(layout, depth):
        if isinstance(layout, awkward1.layout.NumpyArray) and layout.ndim != 1:
            return lambda: layout.toRegularArray()
        elif isinstance(layout, awkward1.layout.EmptyArray) and not allow_empty:
            return lambda: layout.toNumpyArray()
        elif isinstance(layout, awkward1.layout.VirtualArray):
            # FIXME: we must transform the Form (replacing inner_shape with
            # RegularForms) and wrap the ArrayGenerator with regularize_numpy
            return lambda: layout
        else:
            return None

    out = awkward1._util.recursively_apply(to_layout(array), getfunction)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def to_arrow(array):
    """
    Args:
        array: Data to convert to an Apache Arrow array.

    Converts an Awkward Array into an Apache Arrow array.

    This produces arrays of type `pyarrow.Array`. You might need to further
    manipulations (using the pyarrow library) to build a `pyarrow.ChunkedArray`,
    a `pyarrow.RecordBatch`, or a `pyarrow.Table`.

    See also #ak.from_arrow.
    """

    import pyarrow

    layout = to_layout(array)

    def recurse(layout, mask=None):
        if isinstance(layout, awkward1.layout.NumpyArray):
            numpy_arr = numpy.asarray(layout)
            length = len(numpy_arr)
            arrow_type = pyarrow.from_numpy_dtype(numpy_arr.dtype)

            if issubclass(numpy_arr.dtype.type, (numpy.bool_, numpy.bool)):
                if len(numpy_arr) % 8 == 0:
                    ready_to_pack = numpy_arr
                else:
                    ready_to_pack = numpy.empty(
                        int(numpy.ceil(len(numpy_arr) / 8.0)) * 8, dtype=numpy_arr.dtype
                    )
                    ready_to_pack[: len(numpy_arr)] = numpy_arr
                    ready_to_pack[len(numpy_arr) :] = 0
                numpy_arr = numpy.packbits(
                    ready_to_pack.reshape(-1, 8)[:, ::-1].reshape(-1)
                )

            if numpy_arr.ndim == 1:
                if mask is not None:
                    return pyarrow.Array.from_buffers(
                        arrow_type,
                        length,
                        [pyarrow.py_buffer(mask), pyarrow.py_buffer(numpy_arr)],
                    )
                else:
                    return pyarrow.Array.from_buffers(
                        arrow_type, length, [None, pyarrow.py_buffer(numpy_arr)]
                    )
            else:
                return pyarrow.Tensor.from_numpy(numpy_arr)

        elif isinstance(layout, awkward1.layout.EmptyArray):
            return pyarrow.Array.from_buffers(pyarrow.float64(), 0, [None, None])

        elif isinstance(layout, awkward1.layout.ListOffsetArray32):
            offsets = numpy.asarray(layout.offsets, dtype=numpy.int32)

            if layout.parameter("__array__") == "bytestring":
                if mask is None:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.binary(),
                        len(offsets) - 1,
                        [
                            None,
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                else:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.binary(),
                        len(offsets) - 1,
                        [
                            pyarrow.py_buffer(mask),
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                return arrow_arr

            if layout.parameter("__array__") == "string":
                if mask is None:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                    )
                else:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                        pyarrow.py_buffer(mask),
                    )
                return arrow_arr

            content_buffer = recurse(layout.content[: offsets[-1]])
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.list_(content_buffer.type),
                    len(offsets) - 1,
                    [None, pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            else:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.list_(content_buffer.type),
                    len(offsets) - 1,
                    [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            return arrow_arr

        elif isinstance(
            layout,
            (awkward1.layout.ListOffsetArray64, awkward1.layout.ListOffsetArrayU32),
        ):
            offsets = numpy.asarray(layout.offsets, dtype=numpy.int64)

            if layout.parameter("__array__") == "bytestring":
                if mask is None:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.large_binary(),
                        len(offsets) - 1,
                        [
                            None,
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                else:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.large_binary(),
                        len(offsets) - 1,
                        [
                            pyarrow.py_buffer(mask),
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                return arrow_arr

            if layout.parameter("__array__") == "string":
                if mask is None:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                    )
                else:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                        pyarrow.py_buffer(mask),
                    )
                return arrow_arr

            content_buffer = recurse(layout.content[: offsets[-1]])
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.large_list(content_buffer.type),
                    len(offsets) - 1,
                    [None, pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            else:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.large_list(content_buffer.type),
                    len(offsets) - 1,
                    [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            return arrow_arr

        elif isinstance(layout, awkward1.layout.RegularArray):
            return recurse(
                layout.broadcast_tooffsets64(layout.compact_offsets64()), mask
            )

        elif isinstance(
            layout,
            (
                awkward1.layout.ListArray32,
                awkward1.layout.ListArrayU32,
                awkward1.layout.ListArray64,
            ),
        ):
            if mask is not None:
                return recurse(
                    layout.broadcast_tooffsets64(layout.compact_offsets64()), mask
                )
            else:
                return recurse(layout.broadcast_tooffsets64(layout.compact_offsets64()))

        elif isinstance(layout, awkward1.layout.RecordArray):
            values = [recurse(x[: len(layout)]) for x in layout.contents]

            min_list_len = min(map(len, values))

            types = pyarrow.struct(
                [
                    pyarrow.field(layout.keys()[i], values[i].type)
                    for i in range(len(values))
                ]
            )

            if mask is not None:
                return pyarrow.Array.from_buffers(
                    types, min_list_len, [pyarrow.py_buffer(mask)], children=values
                )
            else:
                return pyarrow.Array.from_buffers(
                    types, min_list_len, [None], children=values
                )

        elif isinstance(
            layout,
            (
                awkward1.layout.UnionArray8_32,
                awkward1.layout.UnionArray8_64,
                awkward1.layout.UnionArray8_U32,
            ),
        ):
            values = [recurse(x) for x in layout.contents]
            types = pyarrow.union(
                [pyarrow.field(str(i), values[i].type) for i in range(len(values))],
                "dense",
                list(range(len(values))),
            )

            if mask is not None:
                return pyarrow.Array.from_buffers(
                    types,
                    len(layout.tags),
                    [
                        pyarrow.py_buffer(mask),
                        pyarrow.py_buffer(numpy.asarray(layout.tags)),
                        pyarrow.py_buffer(
                            numpy.asarray(layout.index).astype(numpy.int32)
                        ),
                    ],
                    children=values,
                )
            else:
                return pyarrow.Array.from_buffers(
                    types,
                    len(layout.tags),
                    [
                        None,
                        pyarrow.py_buffer(numpy.asarray(layout.tags)),
                        pyarrow.py_buffer(
                            numpy.asarray(layout.index).astype(numpy.int32)
                        ),
                    ],
                    children=values,
                )

        elif isinstance(
            layout,
            (
                awkward1.layout.IndexedArray32,
                awkward1.layout.IndexedArrayU32,
                awkward1.layout.IndexedArray64,
            ),
        ):
            index = numpy.asarray(layout.index)
            dictionary = recurse(layout.content)
            if mask is None:
                return pyarrow.DictionaryArray.from_arrays(index, dictionary)
            else:
                bytemask = (
                    numpy.unpackbits(~mask)
                    .reshape(-1, 8)[:, ::-1]
                    .reshape(-1)
                    .view(numpy.bool_)
                )[: len(index)]
                return pyarrow.DictionaryArray.from_arrays(index, dictionary, bytemask)

        elif isinstance(
            layout,
            (
                awkward1.layout.IndexedOptionArray32,
                awkward1.layout.IndexedOptionArray64,
            ),
        ):
            index = numpy.array(layout.index, copy=True)
            nulls = index < 0
            index[nulls] = 0

            if len(nulls) % 8 == 0:
                this_bytemask = (~nulls).view(numpy.uint8)
            else:
                length = int(numpy.ceil(len(nulls) / 8.0)) * 8
                this_bytemask = numpy.empty(length, dtype=numpy.uint8)
                this_bytemask[: len(nulls)] = ~nulls
                this_bytemask[len(nulls) :] = 0

            this_bitmask = numpy.packbits(
                this_bytemask.reshape(-1, 8)[:, ::-1].reshape(-1)
            )

            if isinstance(layout, awkward1.layout.IndexedOptionArray32):
                next = awkward1.layout.IndexedArray32(
                    awkward1.layout.Index32(index), layout.content
                )
            else:
                next = awkward1.layout.IndexedArray64(
                    awkward1.layout.Index64(index), layout.content
                )

            if mask is None:
                return recurse(next, this_bitmask)
            else:
                return recurse(next, mask & this_bitmask)

        elif isinstance(layout, awkward1.layout.BitMaskedArray):
            bitmask = numpy.asarray(layout.mask, dtype=numpy.uint8)

            if layout.lsb_order is False:
                bitmask = numpy.packbits(
                    numpy.unpackbits(bitmask).reshape(-1, 8)[:, ::-1].reshape(-1)
                )

            if layout.valid_when is False:
                bitmask = ~bitmask

            return recurse(layout.content[: len(layout)], bitmask).slice(
                length=min(len(bitmask) * 8, len(layout.content))
            )

        elif isinstance(layout, awkward1.layout.ByteMaskedArray):
            mask = numpy.asarray(layout.mask, dtype=numpy.bool) == layout.valid_when

            bytemask = numpy.zeros(
                8 * math.ceil(len(layout.content) / 8), dtype=numpy.bool
            )
            bytemask[: len(mask)] = mask
            bitmask = numpy.packbits(bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

            return recurse(layout.content[: len(layout)], bitmask).slice(
                length=len(mask)
            )

        elif isinstance(layout, (awkward1.layout.UnmaskedArray)):
            return recurse(layout.content)

        else:
            raise TypeError("unrecognized array type: {0}".format(repr(layout)))

    return recurse(layout)


def from_arrow(array, highlevel=True, behavior=None):
    """
    Args:
        array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`,
            or `pyarrow.Table`): Apache Arrow array to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Converts an Apache Arrow array into an Awkward Array.

    See also #ak.to_arrow.
    """

    import pyarrow

    def popbuffers(array, tpe, buffers, length):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            index = popbuffers(
                None if array is None else array.indices,
                tpe.index_type,
                buffers,
                length,
            )
            if array is not None:
                content = recurse(array.dictionary)
            else:
                raise NotImplementedError("Arrow dictionary inside of UnionArray")

            if isinstance(index, awkward1.layout.BitMaskedArray):
                return awkward1.layout.BitMaskedArray(
                    index.mask,
                    awkward1.layout.IndexedArray32(
                        awkward1.layout.Index32(index.content), content
                    ),
                    True,
                    length,
                    True,
                )
            else:
                return awkward1.layout.IndexedArray32(
                    awkward1.layout.Index32(index), content
                )

        elif isinstance(tpe, pyarrow.lib.StructType):
            assert tpe.num_buffers == 1
            mask = buffers.pop(0)
            child_arrays = []
            keys = []
            for i in range(tpe.num_children):
                child_arrays.append(
                    popbuffers(
                        None if array is None else array.field(tpe[i].name),
                        tpe[i].type,
                        buffers,
                        length,
                    )
                )
                keys.append(tpe[i].name)

            out = awkward1.layout.RecordArray(child_arrays, keys)
            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = awkward1.layout.Index32(
                numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)[: length + 1]
            )
            content = popbuffers(
                None if array is None else array.flatten(),
                tpe.value_type,
                buffers,
                offsets[-1],
            )

            out = awkward1.layout.ListOffsetArray32(offsets, content)
            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.LargeListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = awkward1.layout.Index64(
                numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)[: length + 1]
            )
            content = popbuffers(
                None if array is None else array.flatten(),
                tpe.value_type,
                buffers,
                offsets[-1],
            )

            out = awkward1.layout.ListOffsetArray64(offsets, content)
            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            tags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)[:length]
            assert buffers.pop(0) is None
            index = numpy.arange(len(tags), dtype=numpy.int32)

            contents = []
            for i in range(tpe.num_children):
                try:
                    sublength = index[tags == i][-1] + 1
                except IndexError:
                    sublength = 0
                contents.append(popbuffers(None, tpe[i].type, buffers, sublength))
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these[-1] + 1]

            tags = awkward1.layout.Index8(tags)
            index = awkward1.layout.Index32(index)
            out = awkward1.layout.UnionArray8_32(tags, index, contents)

            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            tags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)[:length]
            index = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)[:length]

            contents = []
            for i in range(tpe.num_children):
                try:
                    sublength = index[tags == i].max() + 1
                except ValueError:
                    sublength = 0
                contents.append(popbuffers(None, tpe[i].type, buffers, sublength))
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these.max() + 1]

            tags = awkward1.layout.Index8(tags)
            index = awkward1.layout.Index32(index)
            out = awkward1.layout.UnionArray8_32(tags, index, contents)
            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif tpe == pyarrow.string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)
            contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)

            offsets = awkward1.layout.Index32(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "char")

            awk_arr = awkward1.layout.ListOffsetArray32(offsets, contents)
            awk_arr.setparameter("__array__", "string")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.large_string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)
            contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)

            offsets = awkward1.layout.Index64(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "char")

            awk_arr = awkward1.layout.ListOffsetArray64(offsets, contents)
            awk_arr.setparameter("__array__", "string")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)
            contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)

            offsets = awkward1.layout.Index32(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "byte")

            awk_arr = awkward1.layout.ListOffsetArray32(offsets, contents)
            awk_arr.setparameter("__array__", "bytestring")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.large_binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=numpy.int64)
            contents = numpy.frombuffer(buffers.pop(0), dtype=numpy.uint8)

            offsets = awkward1.layout.Index64(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "byte")

            awk_arr = awkward1.layout.ListOffsetArray64(offsets, contents)
            awk_arr.setparameter("__array__", "bytestring")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.bool_():
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            data = buffers.pop(0)
            out = numpy.frombuffer(data, dtype=numpy.uint8)
            out = numpy.unpackbits(out).reshape(-1, 8)[:, ::-1].reshape(-1)
            out = awkward1.layout.NumpyArray(out[:length])
            if mask is not None:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                mask = numpy.frombuffer(mask, dtype=numpy.uint8)
                return awkward1.layout.BitMaskedArray(awk_mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.DataType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            out = awkward1.layout.NumpyArray(
                numpy.frombuffer(buffers.pop(0), dtype=tpe.to_pandas_dtype())[:length]
            )
            if mask is not None:
                mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=numpy.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        else:
            raise TypeError("unrecognized Arrow array type: {0}".format(repr(tpe)))

    def recurse(obj):
        if isinstance(obj, pyarrow.lib.Array):
            buffers = obj.buffers()
            out = popbuffers(obj, obj.type, buffers, len(obj))
            assert len(buffers) == 0
            return out

        elif isinstance(obj, pyarrow.lib.ChunkedArray):
            chunks = [x for x in obj.chunks if len(x) > 0]
            if len(chunks) == 1:
                return recurse(chunks[0])
            else:
                return awkward1.operations.structure.concatenate(
                    [recurse(x) for x in chunks], highlevel=False
                )

        elif isinstance(obj, pyarrow.lib.RecordBatch):
            child_array = [recurse(obj.column(x)) for x in range(obj.num_columns)]
            keys = obj.schema.names
            awk_arr = awkward1.layout.RecordArray(child_array, keys)
            return awk_arr

        elif isinstance(obj, pyarrow.lib.Table):
            chunks = []
            for batch in obj.to_batches():
                chunk = recurse(batch)
                if len(chunk) > 0:
                    chunks.append(chunk)
            if len(chunks) == 1:
                return chunks[0]
            else:
                return awkward1.operations.structure.concatenate(
                    chunks, highlevel=False
                )

        else:
            raise TypeError("unrecognized Arrow type: {0}".format(type(obj)))

    if highlevel:
        return awkward1._util.wrap(recurse(array), behavior)
    else:
        return recurse(array)


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_")
    and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")
]
