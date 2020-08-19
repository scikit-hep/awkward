# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers
import json
import collections
import math
import threading

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward1.layout
import awkward1._ext
import awkward1._util
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()
numpy = awkward1.nplike.Numpy.instance()


def from_numpy(
    array,
    regulararray=False,
    recordarray=True,
    highlevel=True,
    behavior=None
):
    """
    Args:
        array (np.ndarray): The NumPy array to convert into an Awkward Array.
            This array can be a np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.layout.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.layout.NumpyArray `format`,
            which some functions do not recognize.
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
        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return awkward1.layout.UnmaskedArray(data)
            else:
                def attach(x):
                    if isinstance(x, awkward1.layout.NumpyArray):
                        return awkward1.layout.UnmaskedArray(x)
                    else:
                        return awkward1.layout.RegularArray(
                            attach(x.content), x.size
                        )
                return attach(data.toRegularArray())
        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return awkward1.layout.ByteMaskedArray(
                awkward1.layout.Index8(mask), data, valid_when=False
            )

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
        layout = awkward1.layout.RecordArray(contents, array.dtype.names)

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

    elif isinstance(array, np.ndarray):
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
                    + awkward1._util.exception_suffix(__file__)
                )
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception:
                raise ValueError(
                    "cannot convert {0} into np.ndarray".format(array)
                    + awkward1._util.exception_suffix(__file__)
                )

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
        mask0 = numpy.asarray(array.bytemask()).view(np.bool_)
        if mask0.any():
            if allow_missing:
                mask = numpy.broadcast_to(
                    mask0.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape
                )
                if isinstance(content, numpy.ma.MaskedArray):
                    mask1 = numpy.ma.getmaskarray(content)
                    mask = mask.copy()
                    mask[~mask0] |= mask1

                data[~mask0] = content
                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ValueError(
                    "to_numpy cannot convert 'None' values to "
                    "np.ma.MaskedArray unless the "
                    "'allow_missing' parameter is set to True"
                    + awkward1._util.exception_suffix(__file__)
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
            raise ValueError(
                "cannot convert {0} into np.ndarray".format(array)
                + awkward1._util.exception_suffix(__file__)
            )
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
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(array))
            + awkward1._util.exception_suffix(__file__)
        )

    elif isinstance(array, Iterable):
        return numpy.asarray(array)

    else:
        raise ValueError(
            "cannot convert {0} into np.ndarray".format(array)
            + awkward1._util.exception_suffix(__file__)
        )


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
            raise ValueError(
                "cannot produce an array from a dict"
                + awkward1._util.exception_suffix(__file__)
            )
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

    elif isinstance(array, np.ndarray):
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
        raise TypeError(
            "unrecognized array type: {0}".format(type(array))
            + awkward1._util.exception_suffix(__file__)
        )


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

    elif isinstance(array, np.ndarray):
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
        raise TypeError(
            "unrecognized array type: {0}".format(repr(array))
            + awkward1._util.exception_suffix(__file__)
        )

    if destination is None:
        return out.tojson(pretty=pretty, maxdecimals=maxdecimals)
    else:
        return out.tojson(
            destination, pretty=pretty, maxdecimals=maxdecimals, buffersize=buffersize
        )


def from_awkward0(
    array,
    keep_layout=False,
    regulararray=False,
    recordarray=True,
    highlevel=True,
    behavior=None
):
    """
    Args:
        array (Awkward 0.x or Awkward 1.x array): Data to convert to Awkward
            1.x.
        keep_layout (bool): If True, stay true to the Awkward 0.x layout,
            ensuring zero-copy; otherwise, allow transformations that copy
            data for more flexibility.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.layout.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.layout.NumpyArray `format`,
            which some functions do not recognize.
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
                        np.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[np.newaxis])
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
                        np.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[np.newaxis])
                else:
                    values.append(awkward1.layout.NumpyArray(numpy.array([x])))
            return awkward1.layout.RecordArray(values)[0]

        elif isinstance(array, numpy.ma.MaskedArray):
            return from_numpy(
                array,
                regulararray=regulararray,
                recordarray=recordarray,
                highlevel=False
            )

        elif isinstance(array, np.ndarray):
            return from_numpy(
                array,
                regulararray=regulararray,
                recordarray=recordarray,
                highlevel=False
            )

        elif isinstance(array, awkward0.JaggedArray):
            # starts, stops, content
            # offsetsaliased(starts, stops)
            startsmax = np.iinfo(array.starts.dtype.type).max
            stopsmax = np.iinfo(array.stops.dtype.type).max
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
            indexmax = np.iinfo(array.index.dtype.type).max
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
            mask = awkward1.layout.Index8(array.mask.view(np.int8).reshape(-1))
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
            mask = awkward1.layout.IndexU8(array.mask.view(np.uint8))
            return awkward1.layout.BitMaskedArray(
                mask,
                recurse(array.content, level + 1),
                valid_when=(not array.maskedwhen),
                length=len(array.content),
                lsb_order=array.lsborder,
            )

        elif isinstance(array, awkward0.IndexedMaskedArray):
            # mask, content, maskedwhen
            indexmax = np.iinfo(array.index.dtype.type).max
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
            indexmax = np.iinfo(array.index.dtype.type).max
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
            if keep_layout:
                raise ValueError(
                    "awkward1.SparseArray hasn't been written (if at all); "
                    "try keep_layout=False"
                    + awkward1._util.exception_suffix(__file__)
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
                    + awkward1._util.exception_suffix(__file__)
                )
            return out

        elif isinstance(array, awkward0.ObjectArray):
            # content, generator, args, kwargs
            if keep_layout:
                raise ValueError(
                    "there isn't (and won't ever be) an awkward1 equivalent "
                    "of awkward0.ObjectArray; try keep_layout=False"
                    + awkward1._util.exception_suffix(__file__)
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
            if keep_layout and level != 0:
                raise ValueError(
                    "awkward1 PartitionedArrays are only allowed "
                    "at the root of a data structure, unlike "
                    "awkward0.ChunkedArray; try keep_layout=False"
                    + awkward1._util.exception_suffix(__file__)
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
                + awkward1._util.exception_suffix(__file__)
            )

        elif isinstance(array, awkward0.VirtualArray):
            # generator, args, kwargs, cache, persistentkey, type, nbytes, persistvirtual
            if keep_layout:
                raise NotImplementedError(
                    "FIXME"
                    + awkward1._util.exception_suffix(__file__)
                )
            else:
                return recurse(array.array, level + 1)

        else:
            raise TypeError(
                "not an awkward0 array: {0}".format(repr(array))
                + awkward1._util.exception_suffix(__file__)
            )

    out = recurse(array, 0)
    if highlevel:
        return awkward1._util.wrap(out, behavior)
    else:
        return out


from_awkward0.int8max = np.iinfo(np.int8).max
from_awkward0.int32max = np.iinfo(np.int32).max
from_awkward0.uint32max = np.iinfo(np.uint32).max
from_awkward0.int64max = np.iinfo(np.int64).max


def to_awkward0(array, keep_layout=False):
    """
    Args:
        array: Data to convert into an Awkward 0.x array.
        keep_layout (bool): If True, stay true to the Awkward 1.x layout,
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
            if keep_layout:
                raise ValueError(
                    "awkward0 has no equivalent of RegularArray; "
                    "try keep_layout=False"
                    + awkward1._util.exception_suffix(__file__)
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
                    + awkward1._util.exception_suffix(__file__)
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
            raise NotImplementedError(
                "FIXME"
                + awkward1._util.exception_suffix(__file__)
            )

        else:
            raise AssertionError(
                "missing converter for {0}".format(type(layout).__name__)
                + awkward1._util.exception_suffix(__file__)
            )

    layout = to_layout(
        array, allow_record=True, allow_other=False, numpytype=(np.generic,)
    )
    return recurse(layout)


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(np.number, np.bool_, np.bool),
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

    elif isinstance(array, np.ndarray):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError(
                "NumPy {0} not allowed".format(repr(array.dtype))
                + awkward1._util.exception_suffix(__file__)
            )
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
        raise TypeError(
            "{0} cannot be converted into an Awkward Array".format(array)
            + awkward1._util.exception_suffix(__file__)
        )

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

    layout = to_layout(array, allow_record=False, allow_other=False)

    def recurse(layout, mask=None):
        if isinstance(layout, awkward1.layout.NumpyArray):
            numpy_arr = numpy.asarray(layout)
            length = len(numpy_arr)
            arrow_type = pyarrow.from_numpy_dtype(numpy_arr.dtype)

            if issubclass(numpy_arr.dtype.type, (np.bool_, np.bool)):
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
            offsets = numpy.asarray(layout.offsets, dtype=np.int32)

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
            offsets = numpy.asarray(layout.offsets)

            if len(offsets) == 0 or numpy.max(offsets) <= np.iinfo(np.int32).max:
                small_layout = awkward1.layout.ListOffsetArray32(
                    awkward1.layout.Index32(offsets.astype(np.int32)),
                    layout.content,
                    parameters=layout.parameters,
                )
                return recurse(small_layout, mask=mask)

            offsets = numpy.asarray(layout.offsets, dtype=np.int64)

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
                            numpy.asarray(layout.index).astype(np.int32)
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
                            numpy.asarray(layout.index).astype(np.int32)
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
                    .view(np.bool_)
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
                this_bytemask = (~nulls).view(np.uint8)
            else:
                length = int(numpy.ceil(len(nulls) / 8.0)) * 8
                this_bytemask = numpy.empty(length, dtype=np.uint8)
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
            bitmask = numpy.asarray(layout.mask, dtype=np.uint8)

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
            mask = numpy.asarray(layout.mask, dtype=np.bool) == layout.valid_when

            bytemask = numpy.zeros(
                8 * math.ceil(len(layout.content) / 8), dtype=np.bool
            )
            bytemask[: len(mask)] = mask
            bitmask = numpy.packbits(bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

            return recurse(layout.content[: len(layout)], bitmask).slice(
                length=len(mask)
            )

        elif isinstance(layout, (awkward1.layout.UnmaskedArray)):
            return recurse(layout.content)

        else:
            raise TypeError(
                "unrecognized array type: {0}".format(repr(layout))
                + awkward1._util.exception_suffix(__file__)
            )

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
                raise NotImplementedError(
                    "Arrow dictionary inside of UnionArray"
                    + awkward1._util.exception_suffix(__file__)
                )

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
            for i in range(tpe.num_fields):
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = awkward1.layout.Index32(
                numpy.frombuffer(buffers.pop(0), dtype=np.int32)[: length + 1]
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.LargeListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = awkward1.layout.Index64(
                numpy.frombuffer(buffers.pop(0), dtype=np.int64)[: length + 1]
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            tags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)[:length]
            index = numpy.arange(len(tags), dtype=np.int32)

            contents = []
            for i in range(tpe.num_fields):
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            tags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)[:length]
            index = numpy.frombuffer(buffers.pop(0), dtype=np.int32)[:length]

            contents = []
            for i in range(tpe.num_fields):
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        elif tpe == pyarrow.string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=np.int32)
            contents = numpy.frombuffer(buffers.pop(0), dtype=np.uint8)

            offsets = awkward1.layout.Index32(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "char")

            awk_arr = awkward1.layout.ListOffsetArray32(offsets, contents)
            awk_arr.setparameter("__array__", "string")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.large_string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=np.int64)
            contents = numpy.frombuffer(buffers.pop(0), dtype=np.uint8)

            offsets = awkward1.layout.Index64(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "char")

            awk_arr = awkward1.layout.ListOffsetArray64(offsets, contents)
            awk_arr.setparameter("__array__", "string")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=np.int32)
            contents = numpy.frombuffer(buffers.pop(0), dtype=np.uint8)

            offsets = awkward1.layout.Index32(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "byte")

            awk_arr = awkward1.layout.ListOffsetArray32(offsets, contents)
            awk_arr.setparameter("__array__", "bytestring")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.large_binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)

            offsets = numpy.frombuffer(buffers.pop(0), dtype=np.int64)
            contents = numpy.frombuffer(buffers.pop(0), dtype=np.uint8)

            offsets = awkward1.layout.Index64(offsets)

            contents = awkward1.layout.NumpyArray(contents)
            contents.setparameter("__array__", "byte")

            awk_arr = awkward1.layout.ListOffsetArray64(offsets, contents)
            awk_arr.setparameter("__array__", "bytestring")

            if mask is None:
                return awk_arr
            else:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(
                    awk_mask, awk_arr, True, len(offsets) - 1, True
                )

        elif tpe == pyarrow.bool_():
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            data = buffers.pop(0)
            out = numpy.frombuffer(data, dtype=np.uint8)
            out = numpy.unpackbits(out).reshape(-1, 8)[:, ::-1].reshape(-1)
            out = awkward1.layout.NumpyArray(out[:length].view(np.bool_))
            if mask is not None:
                awk_mask = awkward1.layout.IndexU8(
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                mask = numpy.frombuffer(mask, dtype=np.uint8)
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
                    numpy.frombuffer(mask, dtype=np.uint8)
                )
                return awkward1.layout.BitMaskedArray(mask, out, True, length, True)
            else:
                return out

        else:
            raise TypeError(
                "unrecognized Arrow array type: {0}".format(repr(tpe))
                + awkward1._util.exception_suffix(__file__)
            )

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
            raise TypeError(
                "unrecognized Arrow type: {0}".format(type(obj))
                + awkward1._util.exception_suffix(__file__)
            )

    if highlevel:
        return awkward1._util.wrap(recurse(array), behavior)
    else:
        return recurse(array)

def to_parquet(array, where, explode_records=False, **options):
    """
    Args:
        array: Data to write to a Parquet file.
        where (str, Path, file-like object): Where to write the Parquet file.
        explode_records (bool): If True, lists of records are written as
            records of lists, so that nested keys become top-level fields
            (which can be zipped when read back).
        options: All other options are passed to pyarrow.parquet.ParquetWriter.
            In particular, if no `schema` is given, a schema is derived from
            the array type.

    Writes an Awkward Array to a Parquet file (through pyarrow).

        >>> array1 = ak.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
        >>> awkward1.to_parquet(array1, "array1.parquet")

    See also #ak.to_arrow, which is used as an intermediate step.
    See also #ak.from_parquet.
    """

    import pyarrow
    import pyarrow.parquet

    options["where"] = where

    def batch_iterator(layout):
        if isinstance(layout, awkward1.partition.PartitionedArray):
            for partition in layout.partitions:
                for x in batch_iterator(partition):
                    yield x
        elif isinstance(layout, awkward1.layout.RecordArray):
            names = layout.keys()
            fields = [to_arrow(layout[name]) for name in names]
            yield pyarrow.RecordBatch.from_arrays(fields, names)
        elif explode_records:
            names = layout.keys()
            fields = [layout[name] for name in names]
            layout = awkward1.layout.RecordArray(fields, names, len(layout))
            for x in batch_iterator(layout):
                yield x
        else:
            yield pyarrow.RecordBatch.from_arrays([to_arrow(layout)], [""])

    layout = to_layout(array, allow_record=False, allow_other=False)
    iterator = batch_iterator(layout)
    first = next(iterator)

    if "schema" not in options:
        options["schema"] = first.schema

    writer = pyarrow.parquet.ParquetWriter(**options)
    writer.write_table(pyarrow.Table.from_batches([first]))

    try:
        while True:
            try:
                record_batch = next(iterator)
            except StopIteration:
                break
            else:
                writer.write_table(pyarrow.Table.from_batches([record_batch]))
    finally:
        writer.close()


class _ParquetState(object):
    def __init__(self, file, use_threads, source, options):
        self.file = file
        self.use_threads = use_threads
        self.source = source
        self.options = options

    def __call__(self, row_group, column):
        as_arrow = self.file.read_row_group(row_group, [column], self.use_threads)
        return from_arrow(as_arrow, highlevel=False)[column]

    def __getstate__(self):
        return {
            "use_threads": self.use_threads,
            "source": self.source,
            "options": self.options,
        }

    def __setstate__(self, state):
        self.use_threads = state["use_threads"]
        self.source = state["source"]
        self.options = state["options"]
        self.file = pyarrow.parquet.ParquetFile(self.source, **self.options)


_from_parquet_key_number = 0
_from_parquet_key_lock = threading.Lock()


def _from_parquet_key():
    global _from_parquet_key_number
    with _from_parquet_key_lock:
        out = _from_parquet_key_number
        _from_parquet_key_number += 1
    return out


def from_parquet(
    source,
    columns=None,
    row_groups=None,
    use_threads=True,
    lazy=False,
    lazy_cache="attach",
    lazy_cache_key=None,
    highlevel=True,
    behavior=None,
    **options
):
    """
    Args:
        source (str, Path, file-like object, pyarrow.NativeFile): Where to
            get the Parquet file.
        columns (None or list of str): If None, read all columns; otherwise,
            read a specified set of columns.
        row_groups (None, int, or list of int): If None, read all row groups;
            otherwise, read a single or list of row groups.
        use_threads (bool): Passed to the pyarrow.parquet.ParquetFile.read
            functions; if True, do multithreaded reading.
        lazy (bool): If True, read columns in row groups on demand (as
            #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
            if the file has more than one row group); if False, read all
            requested data immediately.
        lazy_cache (None, "attach", or MutableMapping): If lazy, pass this
            cache to the VirtualArrays. If "attach", a new dict is created
            and attached to the output array as a "cache" parameter on
            #ak.Array.
        lazy_cache_key (None or str): If lazy, pass this cache_key to the
            VirtualArrays. If None, a process-unique string is constructed.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.
        options: All other options are passed to pyarrow.parquet.ParquetFile.

    Reads a Parquet file into an Awkward Array (through pyarrow).

        >>> ak.from_parquet("array1.parquet")
        <Array [[1, 2, 3], [], ... [], [6, 7, 8, 9]] type='6 * var * ?int64'>

    See also #ak.from_arrow, which is used as an intermediate step.
    See also #ak.to_parquet.
    """

    import pyarrow
    import pyarrow.parquet

    file = pyarrow.parquet.ParquetFile(source, **options)
    schema = file.schema.to_arrow_schema()
    all_columns = schema.names

    if columns is None:
        columns = all_columns
    for x in columns:
        if x not in all_columns:
            raise ValueError(
                "column {0} does not exist in file {1}".format(repr(x), repr(source))
                + awkward1._util.exception_suffix(__file__)
            )

    if file.num_row_groups == 0:
        out = awkward1.layout.RecordArray(
            [awkward1.layout.EmptyArray() for x in columns], columns, 0
        )
        if highlevel:
            return awkward1._util.wrap(out, behavior)
        else:
            return out

    if lazy:
        state = _ParquetState(file, use_threads, source, options)

        if lazy_cache == "attach":
            lazy_cache = {}
            toattach = lazy_cache
        else:
            toattach = None

        if lazy_cache is None:
            cache = None
        else:
            cache = awkward1.layout.ArrayCache(lazy_cache)

        if lazy_cache_key is None:
            lazy_cache_key = "ak.from_parquet:{0}".format(_from_parquet_key())

        partitions = []
        offsets = [0]
        for row_group in range(file.num_row_groups):
            length = file.metadata.row_group(row_group).num_rows
            offsets.append(offsets[-1] + length)

            fields = []
            for column in columns:
                generator = awkward1.layout.ArrayGenerator(
                    state,
                    (row_group, column),
                    # form=???      # FIXME: need Arrow schema -> Awkward Forms
                    length=length,
                )
                if all_columns == [""]:
                    cache_key = "{0}[{1}]".format(lazy_cache_key, row_group)
                else:
                    cache_key = "{0}.{1}[{2}]".format(lazy_cache_key, column, row_group)
                fields.append(awkward1.layout.VirtualArray(generator, cache, cache_key))

            if all_columns == [""]:
                partitions.append(fields[0])
            else:
                record = awkward1.layout.RecordArray(fields, columns, length)
                partitions.append(record)

        if len(partitions) == 1:
            out = partitions[0]
        else:
            out = awkward1.partition.IrregularlyPartitionedArray(
                partitions, offsets[1:]
            )
        if highlevel:
            return awkward1._util.wrap(out, behavior, cache=toattach)
        else:
            return out

    else:
        out = from_arrow(
            file.read(columns, use_threads=use_threads),
            highlevel=highlevel,
            behavior=behavior,
        )
        if all_columns == [""]:
            return out[""]
        else:
            return out


def _arrayset_key(
    form_key,
    attribute,
    partition,
    prefix,
    sep,
    partition_first,
):
    if form_key is None:
        raise ValueError(
            "cannot read from arrayset using Forms without form_keys"
            + awkward1._util.exception_suffix(__file__)
        )
    if attribute is None:
        attribute = ""
    else:
        attribute = sep + attribute
    if partition is None:
        return "{0}{1}{2}".format(
            prefix,
            form_key,
            attribute,
        )
    elif partition_first:
        return "{0}{1}{2}{3}{4}".format(
            prefix,
            partition,
            sep,
            form_key,
            attribute,
        )
    else:
        return "{0}{1}{2}{3}{4}".format(
            prefix,
            form_key,
            attribute,
            sep,
            partition,
        )


def to_arrayset(
    array,
    container=None,
    partition=None,
    prefix=None,
    node_format="node{0}",
    partition_format="part{0}",
    sep="-",
    partition_first=False,
):
    u"""
    Args:
        array: Data to decompose into an arrayset.
        container (None or MutableMapping): The str \u2192 NumPy arrays (or
            Python buffers) that represent the decomposed Awkward Array. This
            `container` is only assumed to have a `__setitem__` method that
            accepts strings as keys.
        partition (None or non-negative int): If None and `array` is not
            partitioned, keys written to the container have no reference to
            partitioning; if an integer and `array` is not partitioned, keys
            use this as their partition number; if `array` is partitioned, the
            `partition` argument must be None and keys are written with the
            array's own internal partition numbers.
        prefix (None or str): If None, keys only contain node and partition
            information; if a string, keys are all prepended by `prefix + sep`.
        node_format (str or callable): Python format string or function
            (returning str) of the node part of keys written to the container
            and the `form_key` values in the output Form. Its only argument
            (`{0}` in the format string) is the node number, unique within the
            `array`.
        partition_format (str or callable): Python format string or function
            (returning str) of the partition part of keys written to the
            container (if any). Its only argument (`{0}` in the format string)
            is the partition number.
        sep (str): Separates the prefix, node part, array attribute (e.g.
            `"starts"`, `"stops"`, `"mask"`), and partition part of the
            keys written to the container.
        partition_first (bool): If True, the partition part appears immediately
            after the prefix (if any); if False, the partition part appears
            at the end of the keys. This can be relevant if the `container`
            is sorted or lookup performance depends on alphabetical order.

    Decomposes an Awkward Array into a Form and a collection of arrays, so
    that data can be losslessly written to file formats and storage devices
    that only understand named arrays (or binary blobs).

    This function returns a 3-tuple:

        (form, container, num_partitions)

    where the `form` is a #ak.forms.Form (which can be converted to JSON
    with `tojson`), the `container` is either the MutableMapping you passed in
    or a new dict containing the NumPy arrays, and `num_partitions` is None
    if `array` was not partitioned or the number of partitions if it was.

    These are also the first three arguments of #ak.from_arrayset, so a full
    round-trip is

        >>> reconstituted = ak.from_arrayset(*ak.to_arrayset(original))

    The `container` argument lets you specify your own MutableMapping, which
    might be an interface to some storage format or device (e.g. h5py). It's
    okay if the `container` drops NumPy's `dtype` and `shape` information,
    leaving raw bytes, since `dtype` and `shape` can be reconstituted from
    the #ak.forms.NumpyForm.

    The `partition` argument lets you fill the `container` one partition at a
    time using unpartitioned arrays.

    The rest of the arguments determine the format of the keys written to the
    `container` (which might be restrictive if it represents a storage device).

    Here is a simple example:

        >>> original = ak.Array([[1, 2, 3], [], [4, 5]])
        >>> form, container, num_partitions = ak.to_arrayset(original)
        >>> form
        {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "l",
                "primitive": "int64",
                "form_key": "node1"
            },
            "form_key": "node0"
        }
        >>> container
        {'node0-offsets': array([0, 3, 3, 5], dtype=int64),
         'node1': array([1, 2, 3, 4, 5])}
        >>> print(num_partitions)
        None

    which may be read back with

        >>> ak.from_arrayset(form, container)
        <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>

    (the third argument of #ak.from_arrayset defaults to None).

    Here is an example of building up a partitioned array:

        >>> container = {}
        >>> form, _, _ = ak.to_arrayset(ak.Array([[1, 2, 3], [], [4, 5]]), container, 0)
        >>> form, _, _ = ak.to_arrayset(ak.Array([[6, 7, 8, 9]]), container, 1)
        >>> form, _, _ = ak.to_arrayset(ak.Array([[], [], []]), container, 2)
        >>> form, _, _ = ak.to_arrayset(ak.Array([[10]]), container, 3)
        >>> form
        {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "l",
                "primitive": "int64",
                "form_key": "node1"
            },
            "form_key": "node0"
        }
        >>> container
        {'node0-offsets-part0': array([0, 3, 3, 5], dtype=int64),
         'node1-part0': array([1, 2, 3, 4, 5]),
         'node0-offsets-part1': array([0, 4], dtype=int64),
         'node1-part1': array([6, 7, 8, 9]),
         'node0-offsets-part2': array([0, 0, 0, 0], dtype=int64),
         'node1-part2': array([], dtype=float64),
         'node0-offsets-part3': array([0, 1], dtype=int64),
         'node1-part3': array([10])}

    The object returned by #ak.from_arrayset is now a partitioned array:

        >>> ak.from_arrayset(form, container, 4)
        <Array [[1, 2, 3], [], [4, ... [], [], [10]] type='8 * var * int64'>
        >>> ak.partitions(ak.from_arrayset(form, container, 4))
        [3, 1, 3, 1]

    Which can also lazily load partitions as they are observed:

        >>> lazy = ak.from_arrayset(form, container, 4, lazy=True, lazy_lengths=[3, 1, 3, 1])
        >>> lazy.cache
        {}
        >>> lazy
        <Array [[1, 2, 3], [], [4, ... [], [], [10]] type='8 * var * int64'>
        >>> len(lazy.cache)
        3
        >>> lazy + 100
        <Array [[101, 102, 103], [], ... [], [], [110]] type='8 * var * int64'>
        >>> len(lazy.cache)
        4

    See also #ak.from_arrayset.
    """
    if container is None:
        container = {}

    def index_form(index):
        if isinstance(index, awkward1.layout.Index64):
            return "i64"
        elif isinstance(index, awkward1.layout.Index32):
            return "i32"
        elif isinstance(index, awkward1.layout.IndexU32):
            return "u32"
        elif isinstance(index, awkward1.layout.Index8):
            return "i8"
        elif isinstance(index, awkward1.layout.IndexU8):
            return "u8"
        else:
            raise AssertionError(
                "unrecognized index: " + repr(index)
                + awkward1._util.exception_suffix(__file__)
            )

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + sep

    if isinstance(node_format, str) or (
        awkward1._util.py27 and isinstance(node_format, awkward1._util.unicode)
    ):
        tmp1 = node_format
        node_format = lambda x: tmp1.format(x)
    if isinstance(partition_format, str) or (
        awkward1._util.py27 and isinstance(
            partition_format, awkward1._util.unicode
        )
    ):
        tmp2 = partition_format
        partition_format = lambda x: tmp2.format(x)

    def key(key_index, attribute, partition):
        if partition is not None:
            partition = partition_format(partition)
        return _arrayset_key(
            node_format(key_index),
            attribute,
            partition,
            prefix,
            sep,
            partition_first,
        )

    num_form_keys = [0]

    def little_endian(array):
        return array.astype(array.dtype.newbyteorder("<"), copy=False)

    def fill(layout, part):
        has_identities = layout.identities is not None
        parameters = layout.parameters
        key_index = num_form_keys[0]
        num_form_keys[0] += 1

        if has_identities:
            raise NotImplementedError(
                "ak.to_arrayset for an array with Identities"
                + awkward1._util.exception_suffix(__file__)
            )

        if isinstance(layout, awkward1.layout.EmptyArray):
            array = numpy.asarray(layout)
            container[key(key_index, None, part)] = little_endian(array)
            return awkward1.forms.EmptyForm(
                has_identities, parameters, node_format(key_index)
            )

        elif isinstance(layout, (
            awkward1.layout.IndexedArray32,
            awkward1.layout.IndexedArrayU32,
            awkward1.layout.IndexedArray64
        )):
            container[key(key_index, "index", part)] = little_endian(
                numpy.asarray(layout.index)
            )
            return awkward1.forms.IndexedForm(
                index_form(layout.index),
                fill(layout.content, part),
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, (
            awkward1.layout.IndexedOptionArray32,
            awkward1.layout.IndexedOptionArray64
        )):
            container[key(key_index, "index", part)] = little_endian(
                numpy.asarray(layout.index)
            )
            return awkward1.forms.IndexedOptionForm(
                index_form(layout.index),
                fill(layout.content, part),
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.ByteMaskedArray):
            container[key(key_index, "mask", part)] = little_endian(
                numpy.asarray(layout.mask)
            )
            return awkward1.forms.ByteMaskedForm(
                index_form(layout.mask),
                fill(layout.content, part),
                layout.valid_when,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.BitMaskedArray):
            container[key(key_index, "mask", part)] = little_endian(
                numpy.asarray(layout.mask)
            )
            return awkward1.forms.BitMaskedForm(
                index_form(layout.mask),
                fill(layout.content, part),
                layout.valid_when,
                layout.lsb_order,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.UnmaskedArray):
            return awkward1.forms.UnmaskedForm(
                fill(layout.content, part),
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, (
            awkward1.layout.ListArray32,
            awkward1.layout.ListArrayU32,
            awkward1.layout.ListArray64
        )):
            container[key(key_index, "starts", part)] = little_endian(
                numpy.asarray(layout.starts)
            )
            container[key(key_index, "stops", part)] = little_endian(
                numpy.asarray(layout.stops)
            )
            return awkward1.forms.ListForm(
                index_form(layout.starts),
                index_form(layout.stops),
                fill(layout.content, part),
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, (
            awkward1.layout.ListOffsetArray32,
            awkward1.layout.ListOffsetArrayU32,
            awkward1.layout.ListOffsetArray64
        )):
            container[key(key_index, "offsets", part)] = little_endian(
                numpy.asarray(layout.offsets)
            )
            return awkward1.forms.ListOffsetForm(
                index_form(layout.offsets),
                fill(layout.content, part),
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.NumpyArray):
            array = numpy.asarray(layout)
            container[key(key_index, None, part)] = little_endian(array)
            form = awkward1.forms.Form.from_numpy(array.dtype)
            return awkward1.forms.NumpyForm(
                layout.shape[1:],
                form.itemsize,
                form.format,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.RecordArray):
            if layout.istuple:
                forms = [fill(x, part) for x in layout.contents]
                keys = None
            else:
                forms = []
                keys = []
                for k in layout.keys():
                    forms.append(fill(layout[k], part))
                    keys.append(k)
            return awkward1.forms.RecordForm(
                forms,
                keys,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.RegularArray):
            return awkward1.forms.RegularForm(
                fill(layout.content, part),
                layout.size,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, (
            awkward1.layout.UnionArray8_32,
            awkward1.layout.UnionArray8_U32,
            awkward1.layout.UnionArray8_64
        )):
            forms = []
            for x in layout.contents:
                forms.append(fill(x, part))
            container[key(key_index, "tags", part)] = little_endian(
                numpy.asarray(layout.tags)
            )
            container[key(key_index, "index", part)] = little_endian(
                numpy.asarray(layout.index)
            )
            return awkward1.forms.UnionForm(
                index_form(layout.tags),
                index_form(layout.index),
                forms,
                has_identities,
                parameters,
                node_format(key_index),
            )

        elif isinstance(layout, awkward1.layout.VirtualArray):
            return fill(layout.array, part)

        else:
            raise AssertionError(
                "unrecognized layout node type: " + str(type(layout))
                + awkward1._util.exception_suffix(__file__)
            )

    layout = to_layout(array, allow_record=False, allow_other=False)

    if isinstance(layout, awkward1.partition.PartitionedArray):
        if partition is not None:
            raise ValueError(
                "array is partitioned; an explicit 'partition' should not be "
                "assigned"
                + awkward1._util.exception_suffix(__file__)
            )
        form = None
        for part, content in enumerate(layout.partitions):
            num_form_keys[0] = 0

            f = fill(content, part)

            if form is None:
                form = f
            elif form != f:
                raise ValueError(
                    """the Form of partition {0}:

    {1}

differs from the first Form:

    {2}""".format(part, f.tojson(True, False), form.tojson(True, False))
                    + awkward1._util.exception_suffix(__file__)
                )

        num_partitions = len(layout.partitions)

    else:
        form = fill(layout, partition)
        num_partitions = None

    return form, container, num_partitions


_index_form_to_dtype = _index_form_to_index = _form_to_layout_class = None


def _form_to_layout(
    form,
    container,
    partition,
    prefix,
    sep,
    partition_first,
    cache=None,
    cache_key=None,
    length=None,
):
    global _index_form_to_dtype, _index_form_to_index, _form_to_layout_class

    if _index_form_to_dtype is None:
        _index_form_to_dtype = {
            "i8": np.dtype("<i1"),
            "u8": np.dtype("<u1"),
            "i32": np.dtype("<i4"),
            "u32": np.dtype("<u4"),
            "i64": np.dtype("<i8"),
        }

        _index_form_to_index = {
            "i8": awkward1.layout.Index8,
            "u8": awkward1.layout.IndexU8,
            "i32": awkward1.layout.Index32,
            "u32": awkward1.layout.IndexU32,
            "i64": awkward1.layout.Index64,
        }

        _form_to_layout_class = {
            (awkward1.forms.IndexedForm, "i32"):
                awkward1.layout.IndexedArray32,
            (awkward1.forms.IndexedForm, "u32"):
                awkward1.layout.IndexedArrayU32,
            (awkward1.forms.IndexedForm, "i64"):
                awkward1.layout.IndexedArray64,

            (awkward1.forms.IndexedOptionForm, "i32"):
                awkward1.layout.IndexedOptionArray32,
            (awkward1.forms.IndexedOptionForm, "i64"):
                awkward1.layout.IndexedOptionArray64,

            (awkward1.forms.ListForm, "i32"):
                awkward1.layout.ListArray32,
            (awkward1.forms.ListForm, "u32"):
                awkward1.layout.ListArrayU32,
            (awkward1.forms.ListForm, "i64"):
                awkward1.layout.ListArray64,

            (awkward1.forms.ListOffsetForm, "i32"):
                awkward1.layout.ListOffsetArray32,
            (awkward1.forms.ListOffsetForm, "u32"):
                awkward1.layout.ListOffsetArrayU32,
            (awkward1.forms.ListOffsetForm, "i64"):
                awkward1.layout.ListOffsetArray64,

            (awkward1.forms.UnionForm, "i32"):
                awkward1.layout.UnionArray8_32,
            (awkward1.forms.UnionForm, "u32"):
                awkward1.layout.UnionArray8_U32,
            (awkward1.forms.UnionForm, "i64"):
                awkward1.layout.UnionArray8_64,
        }


    if form.has_identities:
        raise NotImplementedError(
            "ak.from_arrayset for an array with Identities"
            + awkward1._util.exception_suffix(__file__)
        )
    else:
        identities = None

    parameters = form.parameters

    if isinstance(form, awkward1.forms.BitMaskedForm):
        raw_mask = container[_arrayset_key(
            form.form_key,
            "mask",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        mask = _index_form_to_index[form.mask](
            raw_mask.view(_index_form_to_dtype[form.mask])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            len(mask),
        )

        return awkward1.layout.BitMaskedArray(
            mask,
            content,
            form.valid_when,
            len(content),
            form.lsb_order,
            identities,
            parameters,
        )

    elif isinstance(form, awkward1.forms.ByteMaskedForm):
        raw_mask = container[_arrayset_key(
            form.form_key,
            "mask",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        mask = _index_form_to_index[form.mask](
            raw_mask.view(_index_form_to_dtype[form.mask])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            len(mask),
        )

        return awkward1.layout.ByteMaskedArray(
            mask, content, form.valid_when, identities, parameters
        )

    elif isinstance(form, awkward1.forms.EmptyForm):
        return awkward1.layout.EmptyArray(identities, parameters)

    elif isinstance(form, awkward1.forms.IndexedForm):
        raw_index = container[_arrayset_key(
            form.form_key,
            "index",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            numpy.max(index) + 1,
        )

        return _form_to_layout_class[type(form), form.index](
            index, content, identities, parameters
        )

    elif isinstance(form, awkward1.forms.IndexedOptionForm):
        raw_index = container[_arrayset_key(
            form.form_key,
            "index",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            numpy.max(index) + 1,
        )

        return _form_to_layout_class[type(form), form.index](
            index, content, identities, parameters
        )

    elif isinstance(form, awkward1.forms.ListForm):
        raw_starts = container[_arrayset_key(
            form.form_key,
            "starts",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        starts = _index_form_to_index[form.starts](
            raw_starts.view(_index_form_to_dtype[form.starts])
        )
        raw_stops = container[_arrayset_key(
            form.form_key,
            "stops",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        stops = _index_form_to_index[form.stops](
            raw_stops.view(_index_form_to_dtype[form.stops])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            stops[-1],
        )

        return _form_to_layout_class[type(form), form.starts](
            starts, stops, content, identities, parameters
        )

    elif isinstance(form, awkward1.forms.ListOffsetForm):
        raw_offsets = container[_arrayset_key(
            form.form_key,
            "offsets",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        offsets = _index_form_to_index[form.offsets](
            raw_offsets.view(_index_form_to_dtype[form.offsets])
        )

        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            offsets[-1],
        )

        return _form_to_layout_class[type(form), form.offsets](
            offsets, content, identities, parameters
        )

    elif isinstance(form, awkward1.forms.NumpyForm):
        raw_array = container[_arrayset_key(
            form.form_key,
            None,
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")

        dtype_inner_shape = form.to_numpy()
        if dtype_inner_shape.subdtype is None:
            dtype, inner_shape = dtype_inner_shape, ()
        else:
            dtype, inner_shape = dtype_inner_shape.subdtype
        shape = (-1,) + inner_shape

        array = raw_array.view(dtype).reshape(shape)

        return awkward1.layout.NumpyArray(array, identities, parameters)

    elif isinstance(form, awkward1.forms.RecordForm):
        contents = []
        minlength = None
        for content_form in form.contents.values():
            content = _form_to_layout(
                content_form,
                container,
                partition,
                prefix,
                sep,
                partition_first,
                cache,
                cache_key,
                length,
            )
            if minlength is None:
                minlength = len(content)
            else:
                minlength = min(minlength, len(content))
            contents.append(content)

        return awkward1.layout.RecordArray(
            contents,
            None if form.istuple else form.contents.keys(),
            minlength,
            identities,
            parameters,
        )

    elif isinstance(form, awkward1.forms.RegularForm):
        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            length,
        )

        return awkward1.layout.RegularArray(
            content, form.size, identities, parameters
        )

    elif isinstance(form, awkward1.forms.UnionForm):
        raw_tags = container[_arrayset_key(
            form.form_key,
            "tags",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        tags = _index_form_to_index[form.tags](
            raw_tags.view(_index_form_to_dtype[form.tags])
        )
        raw_index = container[_arrayset_key(
            form.form_key,
            "index",
            partition,
            prefix,
            sep,
            partition_first,
        )].reshape(-1).view("u1")
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        contents = []
        for i, x in enumerate(form.contents):
            applicable_indices = numpy.array(index)[numpy.equal(tags, i)]
            contents.append(_form_to_layout(
                x,
                container,
                partition,
                prefix,
                sep,
                partition_first,
                cache,
                cache_key,
                numpy.max(applicable_indices) + 1,
            ))

        return _form_to_layout_class[type(form), form.index](
            tags, index, contents, identities, parameters
        )

    elif isinstance(form, awkward1.forms.UnmaskedForm):
        content = _form_to_layout(
            form.content,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            length,
        )

        return awkward1.layout.UnmaskedArray(content, identities, parameters)

    elif isinstance(form, awkward1.forms.VirtualForm):
        args = (
            form.form,
            container,
            partition,
            prefix,
            sep,
            partition_first,
            cache,
            cache_key,
            length,
        )
        generator = awkward1.layout.ArrayGenerator(
            _form_to_layout,
            args,
            form=form.form,
            length=length,
        )
        node_cache_key = _arrayset_key(
            form.form.form_key,
            "virtual",
            partition,
            prefix,
            sep,
            partition_first,
        )
        return awkward1.layout.VirtualArray(generator, cache, cache_key + sep + node_cache_key)

    else:
        raise AssertionError(
            "unexpected form node type: " + str(type(form))
            + awkward1._util.exception_suffix(__file__)
        )


_from_arrayset_key_number = 0
_from_arrayset_key_lock = threading.Lock()


def _from_arrayset_key():
    global _from_arrayset_key_number
    with _from_arrayset_key_lock:
        out = _from_arrayset_key_number
        _from_arrayset_key_number += 1
    return out


def _wrap_record_with_virtual(input_form):
    def modify(form):
        if form["class"] == "RecordArray":
            for item in form["contents"].values():
                modify(item)
        elif form["class"].startswith("UnionArray"):
            for item in form["contents"]:
                modify(item)
        elif "content" in form:
            modify(form["content"])

        if form["class"] == "RecordArray":
            for key in form["contents"].keys():
                form["contents"][key] = {
                    "class": "VirtualArray",
                    "has_length": True,
                    "form": form["contents"][key],
                }

    form = json.loads(input_form.tojson())
    modify(form)
    return awkward1.forms.Form.fromjson(json.dumps(form))


def from_arrayset(
    form,
    container,
    num_partitions=None,
    prefix=None,
    partition_format="part{0}",
    sep="-",
    partition_first=False,
    lazy=False,
    lazy_cache="attach",
    lazy_cache_key=None,
    lazy_lengths=None,
    highlevel=True,
    behavior=None,
):
    u"""
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to reconstitute from a set of NumPy arrays (or binary blobs).
        container (Mapping, such as dict): The str \u2192 NumPy arrays (or Python
            buffers) that represent the decomposed Awkward Array. This `container`
            is only assumed to have a `__getitem__` method that accepts strings
            as keys.
        num_partitions (None or int): If None, keys are assumed to not describe
            partitions and the return value is an unpartitioned array; if an int,
            this is the number of partitions to look for in the `container`.
        prefix (None or str): If None, keys are assumed to only contain node and
            partition information; if a string, keys are assumed to be prepended
            by `prefix + sep`.
        partition_format (str or callable): Python format string or function
            (returning str) of the partition part of keys in the container (if
            any). Its only argument (`{0}` in the format string) is the
            partition number.
        sep (str): Separates the prefix, node part, array attribute (e.g.
            `"starts"`, `"stops"`, `"mask"`), and partition part of the
            keys expected in the container.
        partition_first (bool): If True, the partition part is assumed to appear
            immediately after the prefix (if any); if False, the partition part
            is assumed to appear at the end of the keys. This can be relevant
            if the `container` is sorted or lookup performance depends on
            alphabetical order.
        lazy (bool): If True, read the array or its partitions on demand (as
            #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
            if `num_partitions` is not None); if False, read all requested data
            immediately. Any RecordArray child nodes will additionally be
            read on demand.
        lazy_cache (None, "attach", or MutableMapping): If lazy, pass this
            cache to the VirtualArrays. If "attach", a new dict is created
            and attached to the output array as a "cache" parameter on
            #ak.Array.
        lazy_cache_key (None or str): If lazy, pass this cache_key to the
            VirtualArrays. If None, a process-unique string is constructed.
        lazy_lengths (None, int, or iterable of ints): If lazy and
            `num_partitions` is None, `lazy_lengths` must be an integer
            specifying the length of the Awkward Array output; if lazy and
            `num_partitions` is an integer, `lazy_lengths` must be an integer
            or iterable of integers specifying the lengths of all partitions.
            This additional input is needed to avoid immediately reading the
            array just to determine its length.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Reconstructs an Awkward Array from a Form and a collection of arrays, so
    that data can be losslessly read from file formats and storage devices that
    only understand named arrays (or binary blobs).

    The first three arguments of this function are the return values of
    #ak.to_arrayset, so a full round-trip is

        >>> reconstituted = ak.from_arrayset(*ak.to_arrayset(original))

    The `container` argument lets you specify your own Mapping, which might be
    an interface to some storage format or device (e.g. h5py). It's okay if
    the `container` dropped NumPy's `dtype` and `shape` information, leaving
    raw bytes, since `dtype` and `shape` can be reconstituted from the
    #ak.forms.NumpyForm.

    The `num_partitions` argument is required for partitioned data, to know how
    many partitions to look for in the `container`.

    The `prefix`, `partition_format`, `sep`, and `partition_first` arguments
    only specify the formatting of the keys, and they have the same meanings as
    in #ak.to_arrayset.

    The arguments that begin with `lazy_` are only needed if `lazy` is True.
    The `lazy_cache` and `lazy_cache_key` determine how the array or its
    partitions are cached after being read from the `container` (in a no-eviction
    dict attached to the output #ak.Array as `cache` if not specified). The
    `lazy_lengths` argument is required.

    See #ak.to_arrayset for examples.
    """

    if isinstance(form, str) or (
        awkward1._util.py27 and isinstance(form, awkward1._util.unicode)
    ):
        form = awkward1.forms.Form.fromjson(form)
    elif isinstance(form, dict):
        form = awkward1.forms.Form.fromjson(json.dumps(form))

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + sep

    if isinstance(partition_format, str) or (
        awkward1._util.py27 and isinstance(
            partition_format, awkward1._util.unicode
        )
    ):
        tmp2 = partition_format
        partition_format = lambda x: tmp2.format(x)

    if lazy:
        form = _wrap_record_with_virtual(form)

        if lazy_cache == "attach":
            lazy_cache = {}
            toattach = lazy_cache
        else:
            toattach = None

        if lazy_cache is not None:
            lazy_cache = awkward1.layout.ArrayCache(lazy_cache)

        if lazy_cache_key is None:
            lazy_cache_key = "ak.from_arrayset:{0}".format(_from_arrayset_key())

    else:
        toattach = None

    if num_partitions is None:
        args = (form, container, None, prefix, sep, partition_first)

        if lazy:
            if not isinstance(lazy_lengths, numbers.Integral):
                raise TypeError(
                    "for lazy=True and num_partitions=None, lazy_lengths "
                    "must be an integer, not " + repr(lazy_lengths)
                    + awkward1._util.exception_suffix(__file__)
                )

            generator = awkward1.layout.ArrayGenerator(
                _form_to_layout,
                args + (lazy_cache, lazy_cache_key, lazy_lengths),
                form=form,
                length=lazy_lengths,
            )

            out = awkward1.layout.VirtualArray(generator, lazy_cache, lazy_cache_key)

        else:
            out = _form_to_layout(*args)

    else:
        if lazy:
            if isinstance(lazy_lengths, numbers.Integral):
                lazy_lengths = [lazy_lengths] * num_partitions
            elif (
                isinstance(lazy_lengths, Iterable) and
                len(lazy_lengths) == num_partitions and
                all(isinstance(x, numbers.Integral) for x in lazy_lengths)
            ):
                pass
            else:
                raise TypeError(
                    "for lazy=True, lazy_lengths must be an integer or "
                    "iterable of 'num_partitions' integers, not "
                    + repr(lazy_lengths)
                    + awkward1._util.exception_suffix(__file__)
                )

        partitions = []
        offsets = [0]

        for part in range(num_partitions):
            p = partition_format(part)
            args = (form, container, p, prefix, sep, partition_first)

            if lazy:
                cache_key = "{0}[{1}]".format(lazy_cache_key, part)

                generator = awkward1.layout.ArrayGenerator(
                    _form_to_layout,
                    args + (lazy_cache, cache_key, lazy_lengths[part]),
                    form=form,
                    length=lazy_lengths[part],
                )

                partitions.append(awkward1.layout.VirtualArray(
                    generator, lazy_cache, cache_key
                ))
                offsets.append(offsets[-1] + lazy_lengths[part])

            else:
                partitions.append(_form_to_layout(*args))
                offsets.append(offsets[-1] + len(partitions[-1]))

        out = awkward1.partition.IrregularlyPartitionedArray(
            partitions, offsets[1:]
        )

    if highlevel:
        return awkward1._util.wrap(out, behavior, cache=toattach)
    else:
        return out


def to_pandas(
    array, how="inner", levelname=lambda i: "sub" * i + "entry", anonymous="values"
):
    """
    Args:
        array: Data to convert into one or more Pandas DataFrames.
        how (None or str): Passed to
            [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
            to combine DataFrames for each multiplicity into one DataFrame. If
            None, a list of Pandas DataFrames is returned.
        levelname (int -> str): Computes a name for each level of the row index
            from the number of levels deep.
        anonymous (str): Column name to use if the `array` does not contain
            records; otherwise, column names are derived from record fields.

    Converts Awkward data structures into Pandas
    [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    rows and columns. The resulting DataFrame(s) contains no Awkward structures.

    #ak.Array structures can't be losslessly converted into a single DataFrame;
    different fields in a record structure might have different nested list
    lengths, but a DataFrame can have only one index.

    If `how` is None, this function always returns a list of DataFrames (even
    if it contains only one DataFrame); otherwise `how` is passed to
    [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
    to merge them into a single DataFrame with the associated loss of data.

    In the following example, nested lists are converted into MultiIndex rows.
    The index level names `"entry"`, `"subentry"` and `"subsubentry"` can be
    controlled with the `levelname` parameter. The column name `"values"` is
    assigned because this array has no fields; it can be controlled with the
    `anonymous` parameter.

        >>> ak.to_pandas(ak.Array([[[1.1, 2.2], [], [3.3]],
        ...                        [],
        ...                        [[4.4], [5.5, 6.6]],
        ...                        [[7.7]],
        ...                        [[8.8]]]))
                                    values
        entry subentry subsubentry
        0     0        0               1.1
                       1               2.2
              2        0               3.3
        2     0        0               4.4
              1        0               5.5
                       1               6.6
        3     0        0               7.7
        4     0        0               8.8

    In this example, nested records are converted into MultiIndex columns.
    (MultiIndex rows and columns can be mixed; these examples are deliberately
    simple.)

        >>> ak.to_pandas(ak.Array([
        ...     {"I": {"a": _, "b": {"i": _}}, "II": {"x": {"y": {"z": _}}}}
        ...     for _ in range(0, 50, 10)]))
                I      II
                a   b   x
                    i   y
                        z
        entry
        0       0   0   0
        1      10  10  10
        2      20  20  20
        3      30  30  30
        4      40  40  40

    The following two examples show how fields of different length lists are
    merged. With `how="inner"` (default), only subentries that exist for all
    fields are preserved; with `how="outer"`, all subentries are preserved at
    the expense of requiring missing values.

        >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                        {"x": [1, 2], "y": [2.2, 1.1]},
        ...                        {"x": [1, 2, 3], "y": [1.1]},
        ...                        {"x": [1, 2, 3, 4], "y": []}]),
        ...                        how="inner")
                        x    y
        entry subentry
        1     0         1  3.3
        2     0         1  2.2
              1         2  1.1
        3     0         1  1.1

    The same with `how="outer"`:

        >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                        {"x": [1, 2], "y": [2.2, 1.1]},
        ...                        {"x": [1, 2, 3], "y": [1.1]},
        ...                        {"x": [1, 2, 3, 4], "y": []}]),
        ...                        how="outer")
                          x    y
        entry subentry
        0     0         NaN  4.4
              1         NaN  3.3
              2         NaN  2.2
              3         NaN  1.1
        1     0         1.0  3.3
              1         NaN  2.2
              2         NaN  1.1
        2     0         1.0  2.2
              1         2.0  1.1
        3     0         1.0  1.1
              1         2.0  NaN
              2         3.0  NaN
        4     0         1.0  NaN
              1         2.0  NaN
              2         3.0  NaN
              3         4.0  NaN
    """
    pandas = awkward1._connect._pandas.get_pandas()

    if how is not None:
        out = None
        for df in to_pandas(array, how=None, levelname=levelname, anonymous=anonymous):
            if out is None:
                out = df
            else:
                out = pandas.merge(out, df, how=how, left_index=True, right_index=True)
        return out

    def recurse(layout, row_arrays, col_names):
        if layout.parameter("__array__") in ("string", "bytestring"):
            return [(to_numpy(layout), row_arrays, col_names)]

        elif layout.purelist_depth > 1:
            offsets, flattened = layout.offsets_and_flatten(axis=1)
            offsets = numpy.asarray(offsets)
            starts, stops = offsets[:-1], offsets[1:]
            counts = stops - starts
            if awkward1._util.win:
                counts = counts.astype(np.int32)
            if len(row_arrays) == 0:
                newrows = [
                    numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
                ]
            else:
                newrows = [numpy.repeat(x, counts) for x in row_arrays]
            newrows.append(
                numpy.arange(offsets[-1], dtype=counts.dtype)
                - numpy.repeat(starts, counts)
            )
            return recurse(flattened, newrows, col_names)

        elif isinstance(layout, awkward1.layout.RecordArray):
            return sum(
                [
                    recurse(layout.field(n), row_arrays, col_names + (n,))
                    for n in layout.keys()
                ],
                [],
            )

        else:
            return [(to_numpy(layout), row_arrays, col_names)]

    layout = to_layout(array, allow_record=True, allow_other=False)
    if isinstance(layout, awkward1.partition.PartitionedArray):
        layout = layout.toContent()

    if isinstance(layout, awkward1.layout.Record):
        layout2 = layout.array[layout.at : layout.at + 1]
    else:
        layout2 = layout

    tables = []
    last_row_arrays = None
    for column, row_arrays, col_names in recurse(layout2, [], ()):
        if isinstance(layout, awkward1.layout.Record):
            row_arrays = row_arrays[1:]  # Record --> one-element RecordArray
        if len(col_names) == 0:
            columns = [anonymous]
        else:
            columns = pandas.MultiIndex.from_tuples([col_names])

        if (
            last_row_arrays is not None
            and len(last_row_arrays) == len(row_arrays)
            and all(
                numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
            )
        ):
            oldcolumns = tables[-1].columns
            if isinstance(oldcolumns, pandas.MultiIndex):
                numold = len(oldcolumns.levels)
            else:
                numold = max(len(x) for x in oldcolumns)
            numnew = len(columns.levels)
            maxnum = max(numold, numnew)
            if numold != maxnum:
                oldcolumns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numold) for x in oldcolumns]
                )
                tables[-1].columns = oldcolumns
            if numnew != maxnum:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numnew) for x in columns]
                )

            newframe = pandas.DataFrame(
                data=column, index=tables[-1].index, columns=columns
            )
            tables[-1] = pandas.concat([tables[-1], newframe], axis=1)

        else:
            if len(row_arrays) == 0:
                index = pandas.RangeIndex(len(column), name=levelname(0))
            else:
                index = pandas.MultiIndex.from_arrays(
                    row_arrays, names=[levelname(i) for i in range(len(row_arrays))]
                )
            tables.append(pandas.DataFrame(data=column, index=index, columns=columns))

        last_row_arrays = row_arrays

    return tables


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_")
    and x not in ("numbers", "json", "Iterable", "numpy", "np", "awkward1")
]
