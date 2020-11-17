# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers
import json
import warnings

try:
    from collections.abc import Iterable
    from collections.abc import MutableMapping
except ImportError:
    from collections import Iterable
    from collections import MutableMapping

import awkward1._util
import awkward1.layout
import awkward1._connect._numpy
import awkward1.operations.convert
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()


@awkward1._connect._numpy.implements("copy")
def copy(array):
    """
    Returns a deep copy of the array (no memory shared with original).

    This is identical to `np.copy` and `copy.deepcopy`.

    It's only useful to explicitly copy an array if you're going to change it
    in-place. This doesn't come up often because Awkward Arrays are immutable.
    That is to say, the Awkward Array library doesn't have any operations that
    change an array in-place, but the data in the array might be owned by another
    library that can change it in-place.

    For example, if the array comes from NumPy:

        >>> underlying_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        >>> wrapper = ak.Array(underlying_array)
        >>> duplicate = ak.copy(wrapper)
        >>> underlying_array[2] = 123
        >>> underlying_array
        array([  1.1,   2.2, 123. ,   4.4,   5.5])
        >>> wrapper
        <Array [1.1, 2.2, 123, 4.4, 5.5] type='5 * float64'>
        >>> duplicate
        <Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>

    There is an exception to this rule: you can add fields to records in an
    #ak.Array in-place. However, this changes the #ak.Array wrapper without
    affecting the underlying layout data (it *replaces* its layout), so a
    shallow copy will do:

        >>> original = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])
        >>> shallow_copy = copy.copy(original)
        >>> shallow_copy["y"] = original.x**2
        >>> shallow_copy
        <Array [{x: 1, y: 1}, ... y: 4}, {x: 3, y: 9}] type='3 * {"x": int64, "y": int64}'>
        >>> original
        <Array [{x: 1}, {x: 2}, {x: 3}] type='3 * {"x": int64}'>

    This is key to Awkward Array's efficiency (memory and speed): operations that
    only change part of a structure re-use pieces from the original ("structural
    sharing"). Changing data in-place would result in many surprising long-distance
    changes, so we don't support it. However, an #ak.Array's data might come from
    a mutable third-party library, so this function allows you to make a true copy.
    """
    layout = awkward1.operations.convert.to_layout(
        array,
        allow_record=True,
        allow_other=False,
    )
    return awkward1._util.wrap(layout.deep_copy(), awkward1._util.behaviorof(array))


def mask(array, mask, valid_when=True, highlevel=True):
    """
    Args:
        array: Data to mask, rather than filter.
        mask (array of booleans): The mask that overlays elements in the
            `array` with None. Must have the same length as `array`.
        valid_when (bool): If True, True values in `mask` are considered
            valid (passed from `array` to the output); if False, False
            values in `mask` are considered valid.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array for which

        output[i] = array[i] if mask[i] == valid_when else None

    Unlike filtering data with #ak.Array.__getitem__, this `output` has the
    same length as the original `array` and can therefore be used in
    calculations with it, such as
    [universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

    For example, with an `array` like

        ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    with a boolean selection of `good` elements like

        >>> good = (array % 2 == 1)
        >>> good
        <Array [False, True, False, ... False, True] type='10 * bool'>

    could be used to filter the original `array` (or another with the same
    length).

        >>> array[good]
        <Array [1, 3, 5, 7, 9] type='5 * int64'>

    However, this eliminates information about which elements were dropped and
    where they were. If we instead use #ak.mask,

        >>> ak.mask(array, good)
        <Array [None, 1, None, 3, ... None, 7, None, 9] type='10 * ?int64'>

    this information and the length of the array is preserved, and it can be
    used in further calculations with the original `array` (or another with
    the same length).

        >>> ak.mask(array, good) + array
        <Array [None, 2, None, 6, ... 14, None, 18] type='10 * ?int64'>

    In particular, successive filters can be applied to the same array.

    Even if the `array` and/or the `mask` is nested,

        >>> array = ak.Array([[[0, 1, 2], [], [3, 4], [5]], [[6, 7, 8], [9]]])
        >>> good = (array % 2 == 1)
        >>> good
        <Array [[[False, True, False], ... [True]]] type='2 * var * var * bool'>

    it can still be used with #ak.mask because the `array` and `mask`
    parameters are broadcasted.

        >>> ak.mask(array, good)
        <Array [[[None, 1, None], ... None], [9]]] type='2 * var * var * ?int64'>

    See #ak.broadcast_arrays for details about broadcasting and the generalized
    set of broadcasting rules.

    Another syntax for

        ak.mask(array, array_of_booleans)

    is

        array.mask[array_of_booleans]

    (which is 5 characters away from simply filtering the `array`).
    """

    def getfunction(inputs, depth):
        layoutarray, layoutmask = inputs
        if isinstance(layoutmask, awkward1.layout.NumpyArray):
            m = awkward1.nplike.of(layoutmask).asarray(layoutmask)
            if not issubclass(m.dtype.type, (np.bool, np.bool_)):
                raise ValueError(
                    "mask must have boolean type, not " "{0}".format(repr(m.dtype))
                    + awkward1._util.exception_suffix(__file__)
                )
            bytemask = awkward1.layout.Index8(m.view(np.int8))
            return lambda: (
                awkward1.layout.ByteMaskedArray(
                    bytemask, layoutarray, valid_when=valid_when
                ).simplify(),
            )
        else:
            return None

    layoutarray = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )
    layoutmask = awkward1.operations.convert.to_layout(
        mask, allow_record=True, allow_other=False
    )

    behavior = awkward1._util.behaviorof(array, mask)
    out = awkward1._util.broadcast_and_apply(
        [layoutarray, layoutmask], getfunction, behavior
    )
    assert isinstance(out, tuple) and len(out) == 1
    if highlevel:
        return awkward1._util.wrap(out[0], behavior)
    else:
        return out[0]


def num(array, axis=1, highlevel=True):
    """
    Args:
        array: Data containing nested lists to count.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array of integers specifying the number of elements at a
    particular level.

    For instance, given the following doubly nested `array`,

        ak.Array([[
                   [1.1, 2.2, 3.3],
                   [],
                   [4.4, 5.5],
                   [6.6]
                  ],
                  [],
                  [
                   [7.7],
                   [8.8, 9.9]]
                  ])

    The number of elements in `axis=1` is

        >>> ak.num(array, axis=1)
        <Array [4, 0, 2] type='3 * int64'>

    and the number of elements at the next level down, `axis=2`, is

        >>> ak.num(array, axis=2)
        <Array [[3, 0, 2, 1], [], [1, 2]] type='3 * var * int64'>

    The `axis=0` case is special: it returns a scalar, the length of the array.

        >>> ak.num(array, axis=0)
        3

    This function is useful for ensuring that slices do not raise errors. For
    instance, suppose that we want to select the first element from each
    of the outermost nested lists of `array`. One of these lists is empty, so
    selecting the first element (`0`) would raise an error. However, if our
    first selection is `ak.num(array) > 0`, we are left with only those lists
    that *do* have a first element:

        >>> array[ak.num(array) > 0, 0]
        <Array [[1.1, 2.2, 3.3], [7.7]] type='2 * var * float64'>

    To keep a placeholder (None) in each place we do not want to select,
    consider using #ak.mask instead of a #ak.Array.__getitem__.

        >>> ak.mask(array, ak.num(array) > 0)[:, 0]
        <Array [[1.1, 2.2, 3.3], None, [7.7]] type='3 * option[var * float64]'>
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    out = layout.num(axis=axis)
    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


def zip(arrays, depth_limit=None, parameters=None, with_name=None, highlevel=True):
    """
    Args:
        arrays (dict or iterable of arrays): Arrays to combine into a
            record-containing structure (if a dict) or a tuple-containing
            structure (if any other kind of iterable).
        depth_limit (None or int): If None, attempt to fully broadcast the
            `array` to all levels. If an int, limit the number of dimensions
            that get broadcasted. The minimum value is `1`, for no
            broadcasting.
        parameters (None or dict): Parameters for the new
            #ak.layout.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.layout.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Combines `arrays` into a single structure as the fields of a collection
    of records or the slots of a collection of tuples. If the `arrays` have
    nested structure, they are broadcasted with one another to form the
    records or tuples as deeply as possible, though this can be limited by
    `depth_limit`.

    This operation may be thought of as the opposite of projection in
    #ak.Array.__getitem__, which extracts fields one at a time, or
    #ak.unzip, which extracts them all in one call.

    Consider the following arrays, `one` and `two`.

        >>> one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
        >>> two = ak.Array([["a", "b", "c"], [], ["d", "e"], ["f"]])

    Zipping them together using a dict creates a collection of records with
    the same nesting structure as `one` and `two`.

        >>> ak.to_list(ak.zip({"x": one, "y": two}))
        [
         [{'x': 1.1, 'y': 'a'}, {'x': 2.2, 'y': 'b'}, {'x': 3.3, 'y': 'c'}],
         [],
         [{'x': 4.4, 'y': 'd'}, {'x': 5.5, 'y': 'e'}],
         [{'x': 6.6, 'y': 'f'}]
        ]

    Doing so with a list creates tuples, whose fields are not named.

        >>> ak.to_list(ak.zip([one, two]))
        [
         [(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
         [],
         [(4.4, 'd'), (5.5, 'e')],
         [(6.6, 'f')]
        ]

    Adding a third array with the same length as `one` and `two` but less
    internal structure is okay: it gets broadcasted to match the others.
    (See #ak.broadcast_arrays for broadcasting rules.)

        >>> three = ak.Array([100, 200, 300, 400])
        >>> ak.to_list(ak.zip([one, two, three]))
        [
         [[(1.1, 97, 100)], [(2.2, 98, 100)], [(3.3, 99, 100)]],
         [],
         [[(4.4, 100, 300)], [(5.5, 101, 300)]],
         [[(6.6, 102, 400)]]
        ]

    However, if arrays have the same depth but different lengths of nested
    lists, attempting to zip them together is a broadcasting error.

        >>> one = ak.Array([[[1, 2, 3], [], [4, 5], [6]], [], [[7, 8]]])
        >>> two = ak.Array([[[1.1, 2.2], [3.3], [4.4], [5.5]], [], [[6.6]]])
        >>> ak.zip([one, two])
        ValueError: in ListArray64, cannot broadcast nested list

    For this, one can set the `depth_limit` to prevent the operation from
    attempting to broadcast what can't be broadcasted.

        >>> ak.to_list(ak.zip([one, two], depth_limit=1))
        [([[1, 2, 3], [], [4, 5], [6]], [[1.1, 2.2], [3.3], [4.4], [5.5]]),
         ([], []),
         ([[7, 8]], [[6.6]])]

    As an extreme, `depth_limit=1` is a handy way to make a record structure
    at the outermost level, regardless of whether the fields have matching
    structure or not.
    """
    if depth_limit is not None and depth_limit <= 0:
        raise ValueError(
            "depth_limit must be None or at least 1"
            + awkward1._util.exception_suffix(__file__)
        )

    if isinstance(arrays, dict):
        recordlookup = []
        layouts = []
        for n, x in arrays.items():
            recordlookup.append(n)
            layouts.append(
                awkward1.operations.convert.to_layout(
                    x, allow_record=False, allow_other=False
                )
            )

    else:
        recordlookup = None
        layouts = []
        for x in arrays:
            layouts.append(
                awkward1.operations.convert.to_layout(
                    x, allow_record=False, allow_other=False
                )
            )

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    def getfunction(inputs, depth):
        if depth_limit == depth or (
            depth_limit is None and
            all(x.purelist_depth == 1 or (
                x.purelist_depth == 2 and
                x.purelist_parameter("__array__") in ("string", "bytestring", "categorical")
            ) for x in inputs)
        ):
            return lambda: (
                awkward1.layout.RecordArray(
                    inputs, recordlookup, parameters=parameters
                ),
            )
        else:
            return None

    behavior = awkward1._util.behaviorof(*arrays)
    out = awkward1._util.broadcast_and_apply(layouts, getfunction, behavior)
    assert isinstance(out, tuple) and len(out) == 1
    if highlevel:
        return awkward1._util.wrap(out[0], behavior)
    else:
        return out[0]


def unzip(array):
    """
    If the `array` contains tuples or records, this operation splits them
    into a Python tuple of arrays, one for each field.

    If the `array` does not contain tuples or records, the single `array`
    is placed in a length 1 Python tuple.

    For example,

        >>> array = ak.Array([{"x": 1.1, "y": [1]},
        ...                   {"x": 2.2, "y": [2, 2]},
        ...                   {"x": 3.3, "y": [3, 3, 3]}])
        >>> x, y = ak.unzip(array)
        >>> x
        <Array [1.1, 2.2, 3.3] type='3 * float64'>
        >>> y
        <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>
    """
    fields = awkward1.operations.describe.fields(array)
    if len(fields) == 0:
        return (array,)
    else:
        return tuple(array[n] for n in fields)


def with_name(array, name, highlevel=True):
    """
    Args:
        base: Data containing records or tuples.
        name (str): Name to give to the records or tuples; this assigns
            the `"__record__"` parameter.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new name. This function does not change the
    array in-place.

    The records or tuples may be nested within multiple levels of nested lists.
    If records are nested within records, only the outermost are affected.

    Setting the `"__record__"` parameter makes it possible to add behaviors
    to the data; see #ak.Array and #ak.behavior for a more complete
    description.
    """

    def getfunction(layout, depth):
        if isinstance(layout, awkward1.layout.RecordArray):
            parameters = dict(layout.parameters)
            parameters["__record__"] = name
            return lambda: awkward1.layout.RecordArray(
                layout.contents,
                layout.recordlookup,
                len(layout),
                layout.identities,
                parameters,
            )
        else:
            return None

    out = awkward1._util.recursively_apply(
        awkward1.operations.convert.to_layout(array), getfunction
    )
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def with_field(base, what, where=None, highlevel=True):
    """
    Args:
        base: Data containing records or tuples.
        what: Data to add as a new field.
        where (None or str or non-empy iterable of str): If None, the new field
            has no name (can be accessed as an integer slot number in a
            string); If str, the name of the new field. If iterable, it is
            interpreted as a path where to add the field in a nested record.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new field attached. This function does not
    change the array in-place.

    See #ak.Array.__setitem__ and #ak.Record.__setitem__ for a variant that
    changes the high-level object in-place. (These methods internally use
    #ak.with_field, so performance is not a factor in choosing one over the
    other.)
    """

    if not (
        where is None
        or isinstance(where, str)
        or (isinstance(where, Iterable) and all(isinstance(x, str) for x in where))
    ):
        raise TypeError(
            "New fields may only be assigned by field name(s) "
            "or as a new integer slot by passing None for 'where'"
            + awkward1._util.exception_suffix(__file__)
        )
    if (
        not isinstance(where, str)
        and isinstance(where, Iterable)
        and all(isinstance(x, str) for x in where)
        and len(where) > 1
    ):
        return with_field(
            base,
            with_field(base[where[0]], what, where=where[1:], highlevel=highlevel),
            where=where[0],
            highlevel=highlevel,
        )
    else:
        if not (isinstance(where, str) or where is None):
            where = where[0]

        behavior = awkward1._util.behaviorof(base, what)
        base = awkward1.operations.convert.to_layout(
            base, allow_record=True, allow_other=False
        )
        if base.numfields < 0:
            raise ValueError(
                "no tuples or records in array; cannot add a new field"
                + awkward1._util.exception_suffix(__file__)
            )

        what = awkward1.operations.convert.to_layout(
            what, allow_record=True, allow_other=True
        )

        def getfunction(inputs, depth):
            nplike = awkward1.nplike.of(*inputs)
            base, what = inputs
            if isinstance(base, awkward1.layout.RecordArray):
                if not isinstance(what, awkward1.layout.Content):
                    what = awkward1.layout.NumpyArray(nplike.repeat(what, len(base)))
                return lambda: (base.setitem_field(where, what),)
            else:
                return None

        keys = base.keys()
        if where in base.keys():
            keys.remove(where)
        if len(keys) == 0:
            # the only key was removed, so just create new Record
            out = (awkward1.layout.RecordArray([what], [where]),)
        else:
            base = base[keys]
            out = awkward1._util.broadcast_and_apply(
                [base, what], getfunction, behavior
            )
        assert isinstance(out, tuple) and len(out) == 1

        if highlevel:
            return awkward1._util.wrap(out[0], behavior=behavior)
        else:
            return out[0]


def with_parameter(array, parameter, value, highlevel=True):
    """
    Args:
        array: Data convertible into an Awkward Array.
        parameter (str): Name of the parameter to set on that array.
        value (JSON): Value of the parameter to set on that array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    This function returns a new array with a parameter set on the outermost
    node of its #ak.Array.layout.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.

    You can also remove a single parameter with this function, since setting
    a parameter to None is equivalent to removing it.
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )

    if isinstance(layout, awkward1.partition.PartitionedArray):
        out = layout.replace_partitions(
            x.withparameter(parameter, value) for x in layout.partitions
        )
    else:
        out = layout.withparameter(parameter, value)

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


def without_parameters(array, highlevel=True):
    """
    Args:
        array: Data convertible into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    This function returns a new array without any parameters in its
    #ak.Array.layout, on nodes of any level of depth.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )

    out = awkward1._util.recursively_apply(
        layout, lambda layout, depth: None, keep_parameters=False
    )

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


_ZEROS = object()


@awkward1._connect._numpy.implements("zeros_like")
def zeros_like(array, highlevel=True):
    """
    Args:
        array: Array to use as a model for a replacement that contains only `0`.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.

    This is the equivalent of NumPy's `np.zeros_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    return full_like(array, _ZEROS, highlevel=highlevel)


@awkward1._connect._numpy.implements("ones_like")
def ones_like(array, highlevel=True):
    """
    Args:
        array: Array to use as a model for a replacement that contains only `1`.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.

    This is the equivalent of NumPy's `np.ones_like` for Awkward Arrays.

    See #ak.full_like for details, and see also #ak.zeros_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    return full_like(array, 1, highlevel=highlevel)


@awkward1._connect._numpy.implements("full_like")
def full_like(array, fill_value, highlevel=True):
    """
    Args:
        array: Array to use as a model for a replacement that contains only
            `fill_value`.
        fill_value: Value to fill new new array with.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.

    This is the equivalent of NumPy's `np.full_like` for Awkward Arrays.

    Although it's possible to produce an array of `fill_value` with the structure
    of an `array` using #ak.broadcast_arrays:

        >>> array = ak.Array([[1, 2, 3], [], [4, 5]])
        >>> ak.broadcast_arrays(array, 1)
        [<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
         <Array [[1, 1, 1], [], [1, 1]] type='3 * var * int64'>]
        >>> ak.broadcast_arrays(array, 1.0)
        [<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
         <Array [[1, 1, 1], [], [1, 1]] type='3 * var * float64'>]

    Such a technique takes its type from the scalar (`1` or `1.0`), rather than
    the array. This function gets all types from the array, which might not be
    the same in all parts of the structure.

    Here is an extreme example:

        >>> array = ak.Array([
        ... [{"x": 0.0, "y": []},
        ...  {"x": 1.1, "y": [1]},
        ...  {"x": 2.2, "y": [1, 2]}],
        ... [],
        ... [{"x": 3.3, "y": [1, 2, None, 3]},
        ...  False,
        ...  False,
        ...  True,
        ...  {"x": 4.4, "y": [1, 2, None, 3, 4]}]])
        >>> ak.to_list(ak.full_like(array, 12.3))
        [[{"x": 12.3, "y": []},
          {"x": 12.3, "y": [12]},
          {"x": 12.3, "y": [12, 12]}],
         [],
         [{"x": 12.3, "y": [12, 12, None, 12]},
          True,
          True,
          True,
          {"x": 12.3, "y": [12, 12, None, 12, 12]}]]

    The `"x"` values get filled in with `12.3` because they retain their type
    (`float64`) and the `"y"` list items get filled in with `12` because they
    retain their type (`int64`). Booleans get filled with True because `12.3`
    is not zero. Missing values remain in the same positions as in the original
    `array`. (To fill them in, use #ak.fill_none.)

    See also #ak.zeros_like and #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )

    def getfunction(layout, depth):
        if layout.parameter("__array__") == "bytestring" and fill_value is _ZEROS:
            nplike = awkward1.nplike.of(layout)
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return lambda: awkward1.layout.ListArray64(
                awkward1.layout.Index64(nplike.zeros(len(layout), dtype=np.int64)),
                awkward1.layout.Index64(nplike.zeros(len(layout), dtype=np.int64)),
                awkward1.layout.NumpyArray(asbytes, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "bytestring":
            nplike = awkward1.nplike.of(layout)
            if isinstance(fill_value, bytes):
                asbytes = fill_value
            elif (
                isinstance(fill_value, str)
                or (awkward1._util.py27 and isinstance(fill_value, awkward1._util.unicode))
            ):
                asbytes = fill_value.encode("utf-8", "surrogateescape")
            else:
                asbytes = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asbytes, dtype=np.uint8)

            return lambda: awkward1.layout.ListArray64(
                awkward1.layout.Index64(
                    nplike.zeros(len(layout), dtype=np.int64)
                ),
                awkward1.layout.Index64(
                    nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                awkward1.layout.NumpyArray(
                    asbytes, parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "string" and fill_value is _ZEROS:
            nplike = awkward1.nplike.of(layout)
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return lambda: awkward1.layout.ListArray64(
                awkward1.layout.Index64(nplike.zeros(len(layout), dtype=np.int64)),
                awkward1.layout.Index64(nplike.zeros(len(layout), dtype=np.int64)),
                awkward1.layout.NumpyArray(asbytes, parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )

        elif layout.parameter("__array__") == "string":
            nplike = awkward1.nplike.of(layout)
            asstr = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asstr, dtype=np.uint8)
            return lambda: awkward1.layout.ListArray64(
                awkward1.layout.Index64(
                    nplike.zeros(len(layout), dtype=np.int64)
                ),
                awkward1.layout.Index64(
                    nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                awkward1.layout.NumpyArray(
                    asbytes, parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
            )

        elif isinstance(layout, awkward1.layout.NumpyArray):
            nplike = awkward1.nplike.of(layout)
            original = nplike.asarray(layout)
            if fill_value == 0 or fill_value is _ZEROS:
                return lambda: awkward1.layout.NumpyArray(
                    nplike.zeros_like(original),
                    layout.identities,
                    layout.parameters,
                )
            elif fill_value == 1:
                return lambda: awkward1.layout.NumpyArray(
                    nplike.ones_like(original),
                    layout.identities,
                    layout.parameters,
                )
            else:
                return lambda: awkward1.layout.NumpyArray(
                    nplike.full_like(original, fill_value),
                    layout.identities,
                    layout.parameters,
                )
        else:
            return None

    out = awkward1._util.recursively_apply(layout, getfunction)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


@awkward1._connect._numpy.implements("broadcast_arrays")
def broadcast_arrays(*arrays, **kwargs):
    """
    Args:
        arrays: Arrays to broadcast into the same structure.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.

    Like NumPy's
    [broadcast_arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_arrays.html)
    function, this function returns the input `arrays` with enough elements
    duplicated that they can be combined element-by-element.

    For NumPy arrays, this means that scalars are replaced with arrays with
    the same scalar value repeated at every element of the array, and regular
    dimensions are created to increase low-dimensional data into
    high-dimensional data.

    For example,

        >>> ak.broadcast_arrays(5,
        ...                     [1, 2, 3, 4, 5])
        [<Array [5, 5, 5, 5, 5] type='5 * int64'>,
         <Array [1, 2, 3, 4, 5] type='5 * int64'>]

    and

        >>> ak.broadcast_arrays(np.array([1, 2, 3]),
        ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
        [<Array [[  1,   2,   3], [ 1,  2,  3]] type='2 * 3 * int64'>,
         <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]

    Note that in the second example, when the `3 * int64` array is expanded
    to match the `2 * 3 * float64` array, it is the deepest dimension that
    is aligned. If we try to match a `2 * int64` with the `2 * 3 * float64`,

        >>> ak.broadcast_arrays(np.array([1, 2]),
        ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
        ValueError: cannot broadcast RegularArray of size 2 with RegularArray of size 3

    NumPy has the same behavior: arrays with different numbers of dimensions
    are aligned to the right before expansion. One can control this by
    explicitly adding a new axis (reshape to add a dimension of length 1)
    where the expansion is supposed to take place because a dimension of
    length 1 can be expanded like a scalar.

        >>> ak.broadcast_arrays(np.array([1, 2])[:, np.newaxis],
        ...                     np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
        [<Array [[  1,   1,   1], [ 2,  2,  2]] type='2 * 3 * int64'>,
         <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]

    Again, NumPy does the same thing (`np.newaxis` is equal to None, so this
    trick is often shown with None in the slice-tuple). Where the broadcasting
    happens can be controlled, but numbers of dimensions that don't match are
    implicitly aligned to the right (fitting innermost structure, not
    outermost).

    While that might be an arbitrary decision for rectilinear arrays, it is
    much more natural for implicit broadcasting to align left for tree-like
    structures. That is, the root of each data structure should agree and
    leaves may be duplicated to match. For example,

        >>> ak.broadcast_arrays([            100,   200,        300],
        ...                     [[1.1, 2.2, 3.3],    [], [4.4, 5.5]])
        [<Array [[100, 100, 100], [], [300, 300]] type='3 * var * int64'>,
         <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>]

    One typically wants single-item-per-element data to be duplicated to
    match multiple-items-per-element data. Operations on the broadcasted
    arrays like

        one_dimensional + nested_lists

    would then have the same effect as the procedural code

        for x, outer in zip(one_dimensional, nested_lists):
            output = []
            for inner in outer:
                output.append(x + inner)
            yield output

    where `x` has the same value for each `inner` in the inner loop.

    Awkward Array's broadcasting manages to have it both ways by applying the
    following rules:

       * If a dimension is regular (i.e. #ak.types.RegularType), like NumPy,
         implicit broadcasting aligns to the right, like NumPy.
       * If a dimension is variable (i.e. #ak.types.ListType), which can
         never be true of NumPy, implicit broadcasting aligns to the left.
       * Explicit broadcasting with a length-1 regular dimension always
         broadcasts, like NumPy.

    Thus, it is important to be aware of the distinction between a dimension
    that is declared to be regular in the type specification and a dimension
    that is allowed to be variable (even if it happens to have the same length
    for all elements). This distinction is can be accessed through the
    #ak.Array.type, but it is lost when converting an array into JSON or
    Python objects.
    """
    (highlevel,) = awkward1._util.extra((), kwargs, [("highlevel", True)])

    inputs = []
    for x in arrays:
        y = awkward1.operations.convert.to_layout(
            x, allow_record=True, allow_other=True
        )
        if isinstance(y, awkward1.partition.PartitionedArray):
            y = y.toContent()
        if not isinstance(y, (awkward1.layout.Content, awkward1.layout.Record)):
            y = awkward1.layout.NumpyArray(awkward1.nplike.of(*arrays).array([y]))
        inputs.append(y)

    def getfunction(inputs, depth):
        if all(isinstance(x, awkward1.layout.NumpyArray) for x in inputs):
            return lambda: tuple(inputs)
        else:
            return None

    behavior = awkward1._util.behaviorof(*arrays)
    out = awkward1._util.broadcast_and_apply(inputs, getfunction, behavior)
    assert isinstance(out, tuple)
    if highlevel:
        return [awkward1._util.wrap(x, behavior) for x in out]
    else:
        return list(out)


@awkward1._connect._numpy.implements("concatenate")
def concatenate(arrays, axis=0, mergebool=True, highlevel=True):
    """
    Args:
        arrays: Arrays to concatenate along any dimension.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        mergebool (bool): If True, boolean and nummeric data can be combined
            into the same buffer, losing information about False vs `0` and
            True vs `1`; otherwise, they are kept in separate buffers with
            distinct types (using an #ak.layout.UnionArray8_64).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array with `arrays` concatenated. For `axis=0`, this means that
    one whole array follows another. For `axis=1`, it means that the `arrays`
    must have the same lengths and nested lists are each concatenated,
    element for element, and similarly for deeper levels.
    """
    contents = [
        awkward1.operations.convert.to_layout(x, allow_record=False if axis == 0 else True) for x in arrays
    ]
    contents = [
        x.toContent() if isinstance(x, awkward1.partition.PartitionedArray) else x
        for x in contents
    ]
    if len(contents) == 0:
        raise ValueError(
            "need at least one array to concatenate"
            + awkward1._util.exception_suffix(__file__)
        )

    posaxis = contents[0].axis_wrap_if_negative(axis)
    if posaxis == 0:
        batch = [contents[0]]
        for x in contents[1:]:
            if batch[-1].mergeable(x, mergebool=mergebool):
                batch.append(x)
            else:
                collapsed = batch[0].mergemany(batch[1:])
                batch = [collapsed.merge_as_union(x)]

        out = batch[0].mergemany(batch[1:])

    else:
        for x in contents[1:]:
            if x.axis_wrap_if_negative(axis) != posaxis:
                raise ValueError(
                    "cannot concatenate arrays in different axis"
                    + awkward1._util.exception_suffix(__file__)
                )

        def getfunction(inputs, depth):
            if depth == posaxis:
                out = inputs[0].mergemany_as_union(inputs[1:], 1)
                return lambda: (out,)
            else:
                return None

        out = awkward1._util.broadcast_and_apply(contents, getfunction,
            behavior=awkward1._util.behaviorof(*arrays), allow_records=True)[0]

    if isinstance(out, awkward1._util.uniontypes):
        out = out.simplify(mergebool=mergebool)

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(*arrays))
    else:
        return out


@awkward1._connect._numpy.implements("where")
def where(condition, *args, **kwargs):
    """
    Args:
        condition (np.ndarray or rectilinear #ak.Array of booleans): In the
            three-argument form of this function (`condition`, `x`, `y`),
            True values in `condition` select values from `x` and False
            values in `condition` select values from `y`.
        x: Data with the same length as `condition`.
        y: Data with the same length as `condition`.
        mergebool (bool, default is True): If True, boolean and nummeric data
            can be combined into the same buffer, losing information about
            False vs `0` and True vs `1`; otherwise, they are kept in separate
            buffers with distinct types (using an #ak.layout.UnionArray8_64).
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.

    This function has a one-argument form, `condition` without `x` or `y`, and
    a three-argument form, `condition`, `x`, and `y`. In the one-argument form,
    it is completely equivalent to NumPy's
    [nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html)
    function.

    In the three-argument form, it acts as a vectorized ternary operator:
    `condition`, `x`, and `y` must all have the same length and

        output[i] = x[i] if condition[i] else y[i]

    for all `i`. The structure of `x` and `y` do not need to be the same; if
    they are incompatible types, the output will have #ak.type.UnionType.
    """
    mergebool, highlevel = awkward1._util.extra(
        (), kwargs, [("mergebool", True), ("highlevel", True)]
    )

    akcondition = awkward1.operations.convert.to_layout(condition, allow_record=False)

    if isinstance(akcondition, awkward1.partition.PartitionedArray):
        akcondition = akcondition.replace_partitions(
            [
                awkward1.layout.NumpyArray(awkward1.operations.convert.to_numpy(x))
                for x in akcondition.partitions
            ]
        )
    else:
        akcondition = awkward1.layout.NumpyArray(
            awkward1.operations.convert.to_numpy(akcondition)
        )

    if len(args) == 0:
        nplike = awkward1.nplike.of(akcondition)
        out = nplike.nonzero(awkward1.operations.convert.to_numpy(akcondition))
        if highlevel:
            return tuple(
                awkward1._util.wrap(
                    awkward1.layout.NumpyArray(x), awkward1._util.behaviorof(condition)
                )
                for x in out
            )
        else:
            return tuple(awkward1.layout.NumpyArray(x) for x in out)

    elif len(args) == 1:
        raise ValueError(
            "either both or neither of x and y should be given"
            + awkward1._util.exception_suffix(__file__)
        )

    elif len(args) == 2:
        nplike = awkward1.nplike.of(akcondition, args[0], args[1])
        x = awkward1.operations.convert.to_layout(args[0], allow_record=False)
        y = awkward1.operations.convert.to_layout(args[1], allow_record=False)

        def do_one(akcondition, x, y):
            tags = nplike.asarray(akcondition) == 0
            assert tags.itemsize == 1
            index = nplike.empty(len(tags), dtype=np.int64)
            index = nplike.arange(len(akcondition), dtype=np.int64)

            tags = awkward1.layout.Index8(tags.view(np.int8))
            index = awkward1.layout.Index64(index)
            tmp = awkward1.layout.UnionArray8_64(tags, index, [x, y])
            return tmp.simplify(mergebool=mergebool)

        sample = None
        if isinstance(akcondition, awkward1.partition.PartitionedArray):
            sample = akcondition
        elif isinstance(x, awkward1.partition.PartitionedArray):
            sample = x
        elif isinstance(y, awkward1.partition.PartitionedArray):
            sample = y

        if sample is not None:
            akcondition, x, y = awkward1.partition.partition_as(
                sample, (akcondition, x, y)
            )
            output = []
            for part in awkward1.partition.iterate(
                sample.numpartitions, (akcondition, x, y)
            ):
                output.append(do_one(*part))

            out = awkward1.partition.IrregularlyPartitionedArray(output)

        else:
            out = do_one(akcondition, x, y)

        return awkward1._util.wrap(
            out, behavior=awkward1._util.behaviorof(condition, *args)
        )

    else:
        raise TypeError(
            "where() takes from 1 to 3 positional arguments but {0} were "
            "given".format(len(args) + 1)
            + awkward1._util.exception_suffix(__file__)
        )


def flatten(array, axis=1, highlevel=True):
    """
    Args:
        array: Data containing nested lists to flatten.
        axis (None or int): If None, the operation flattens all levels of
            nesting, returning a 1-dimensional array. Otherwise, it flattens
            at a specified depth. The outermost dimension is `0`, followed
            by `1`, etc., and negative values count backward from the
            innermost: `-1` is the innermost dimension, `-2` is the next
            level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array with one level of nesting removed by erasing the
    boundaries between consecutive lists. Since this operates on a level of
    nesting, `axis=0` is a special case that only removes values at the
    top level that are equal to None.

    Consider the following doubly nested `array`.

        ak.Array([[
                   [1.1, 2.2, 3.3],
                   [],
                   [4.4, 5.5],
                   [6.6]],
                  [],
                  [
                   [7.7],
                   [8.8, 9.9]
                  ]])

    At `axis=1`, the outer lists (length 4, length 0, length 2) become a single
    list (of length 6).

        >>> print(ak.flatten(array, axis=1))
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7], [8.8, 9.9]]

    At `axis=2`, the inner lists (lengths 3, 0, 2, 1, 1, and 2) become three
    lists (of lengths 6, 0, and 3).

        >>> print(ak.flatten(array, axis=2))
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [], [7.7, 8.8, 9.9]]

    There's also an option to completely flatten the array with `axis=None`.
    This is useful for passing the data to a function that doesn't care about
    nested structure, such as a plotting routine.

        >>> print(ak.flatten(array, axis=None))
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    Missing values are eliminated by flattening: there is no distinction
    between an empty list and a value of None at the level of flattening.

        >>> array = ak.Array([[1.1, 2.2, 3.3], None, [4.4], [], [5.5]])
        >>> ak.flatten(array, axis=1)
        <Array [1.1, 2.2, 3.3, 4.4, 5.5] type='5 * float64'>

    As a consequence, flattening at `axis=0` does only one thing: it removes
    None values from the top level.

        >>> ak.flatten(array, axis=0)
        <Array [[1.1, 2.2, 3.3], [4.4], [], [5.5]] type='4 * var * float64'>

    As a technical detail, the flattening operation can be trivial in a common
    case, #ak.layout.ListOffsetArray in which the first `offset` is `0`.
    In that case, the flattened data is simply the array node's `content`.

        >>> array.layout
        <ListOffsetArray64>
            <offsets><Index64 i="[0 4 4 6]" offset="0" length="4"/></offsets>
            <content><ListOffsetArray64>
                <offsets><Index64 i="[0 3 3 5 6 7 9]" offset="0" length="7"/></offsets>
                <content>
                    <NumpyArray format="d" shape="9" data="1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9"/>
                </content>
            </ListOffsetArray64></content>
        </ListOffsetArray64>
        >>> np.asarray(array.layout.content.content)
        array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

    However, it is important to keep in mind that this is a special case:
    #ak.flatten and `content` are not interchangeable!
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    nplike = awkward1.nplike.of(layout)

    if axis is None:
        out = awkward1._util.completely_flatten(layout)
        assert isinstance(out, tuple) and all(isinstance(x, np.ndarray) for x in out)

        if any(isinstance(x, nplike.ma.MaskedArray) for x in out):
            out = awkward1.layout.NumpyArray(nplike.ma.concatenate(out))
        else:
            out = awkward1.layout.NumpyArray(nplike.concatenate(out))

    elif axis == 0:

        def apply(layout):
            if isinstance(layout, awkward1._util.virtualtypes):
                return apply(layout.array)

            elif isinstance(layout, awkward1._util.unknowntypes):
                return apply(awkward1.layout.NumpyArray(nplike.array([])))

            elif isinstance(layout, awkward1._util.indexedtypes):
                return apply(layout.project())

            elif isinstance(layout, awkward1._util.uniontypes):
                if not any(
                    isinstance(x, awkward1._util.optiontypes)
                    and not isinstance(x, awkward1.layout.UnmaskedArray)
                    for x in layout.contents
                ):
                    return layout

                tags = nplike.asarray(layout.tags)
                index = nplike.array(nplike.asarray(layout.index), copy=True)
                bigmask = nplike.empty(len(index), dtype=np.bool_)
                for tag, content in enumerate(layout.contents):
                    if isinstance(
                        content, awkward1._util.optiontypes
                    ) and not isinstance(content, awkward1.layout.UnmaskedArray):
                        bigmask[:] = False
                        bigmask[tags == tag] = nplike.asarray(
                            content.bytemask()
                        ).view(np.bool_)
                        index[bigmask] = -1

                good = index >= 0
                return awkward1.layout.UnionArray8_64(
                    awkward1.layout.Index8(tags[good]),
                    awkward1.layout.Index64(index[good]),
                    layout.contents,
                )

            elif isinstance(layout, awkward1._util.optiontypes):
                return layout.project()

            else:
                return layout

        if isinstance(layout, awkward1.partition.PartitionedArray):
            out = awkward1.partition.IrregularlyPartitionedArray(
                [apply(x) for x in layout.partitions]
            )
        else:
            out = apply(layout)

        if highlevel:
            return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
        else:
            return out

    else:
        out = layout.flatten(axis)

    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def local_index(array, axis=-1, highlevel=True):
    """
    Args:
        array: Array to index.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    For example,

        >>> array = ak.Array([
        ...     [[0.0, 1.1, 2.2], []],
        ...     [[3.3, 4.4]],
        ...     [],
        ...     [[5.5], [], [6.6, 7.7, 8.8, 9.9]]])
        >>> ak.local_index(array, axis=0)
        <Array [0, 1, 2, 3] type='4 * int64'>
        >>> ak.local_index(array, axis=1)
        <Array [[0, 1], [0], [], [0, 1, 2]] type='4 * var * int64'>
        >>> ak.local_index(array, axis=2)
        <Array [[[0, 1, 2], []], ... [], [0, 1, 2, 3]]] type='4 * var * var * int64'>

    Note that you can make a Pandas-style MultiIndex by calling this function on
    every axis.

        >>> multiindex = ak.zip([ak.local_index(array, i) for i in range(array.ndim)])
        >>> multiindex
        <Array [[[(0, 0, 0), (0, 0, ... ), (3, 2, 3)]]] type='4 * var * var * (int64, in...'>
        >>> ak.to_list(multiindex)
        [[[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
         [[(1, 0, 0), (1, 0, 1)]],
         [],
         [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]]]
        >>> ak.to_list(ak.flatten(ak.flatten(multiindex)))
        [(0, 0, 0),
         (0, 0, 1),
         (0, 0, 2),
         (1, 0, 0),
         (1, 0, 1),
         (3, 0, 0),
         (3, 2, 0),
         (3, 2, 1),
         (3, 2, 2),
         (3, 2, 3)]

    But if you're interested in Pandas, you may want to use #ak.to_pandas directly.

        >>> ak.to_pandas(array)
                                    values
        entry subentry subsubentry
        0     0        0               0.0
                       1               1.1
                       2               2.2
        1     0        0               3.3
                       1               4.4
        3     0        0               5.5
              2        0               6.6
                       1               7.7
                       2               8.8
                       3               9.9
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )
    out = layout.localindex(axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


@awkward1._connect._numpy.implements("sort")
def sort(array, axis=-1, ascending=True, stable=True, highlevel=True):
    """
    Args:
        array: Data to sort, possibly within nested lists.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        ascending (bool): If True, the first value in each sorted group
            will be smallest, the last value largest; if False, the order
            is from largest to smallest.
        stable (bool): If True, use a stable sorting algorithm (introsort:
            a hybrid of quicksort, heapsort, and insertion sort); if False,
            use a sorting algorithm that is not guaranteed to be stable
            (heapsort).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    For example,

        >>> ak.sort(ak.Array([[7, 5, 7], [], [2], [8, 2]]))
        <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    out = layout.sort(axis, ascending, stable)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


@awkward1._connect._numpy.implements("argsort")
def argsort(array, axis=-1, ascending=True, stable=True, highlevel=True):
    """
    Args:
        array: Data for which to get a sorting index, possibly within nested
            lists.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        ascending (bool): If True, the first value in each sorted group
            will be smallest, the last value largest; if False, the order
            is from largest to smallest.
        stable (bool): If True, use a stable sorting algorithm (introsort:
            a hybrid of quicksort, heapsort, and insertion sort); if False,
            use a sorting algorithm that is not guaranteed to be stable
            (heapsort).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    For example,

        >>> ak.argsort(ak.Array([[7.7, 5.5, 7.7], [], [2.2], [8.8, 2.2]]))
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>

    The result of this function can be used to index other arrays with the
    same shape:

        >>> data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
        >>> index = ak.argsort(index)
        >>> index
        <Array [[1, 0, 2], [], [0], [1, 0]] type='4 * var * int64'>
        >>> data[index]
        <Array [[5, 7, 7], [], [2], [2, 8]] type='4 * var * int64'>
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    out = layout.argsort(axis, ascending, stable)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def pad_none(array, target, axis=1, clip=False, highlevel=True):
    """
    Args:
        array: Data containing nested lists to pad to a target length.
        target (int): The intended length of the lists. If `clip=True`,
            the output lists will have exactly this length; otherwise,
            they will have *at least* this length.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        clip (bool): If True, the output lists will have regular lengths
            (#ak.types.RegularType) of exactly `target`; otherwise the
            output lists will have in-principle variable lengths
            (#ak.types.ListType) of at least `target`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Increase the lengths of lists to a target length by adding None values.

    Consider the following doubly nested `array`.

        ak.Array([[
                   [1.1, 2.2, 3.3],
                   [],
                   [4.4, 5.5],
                   [6.6]],
                  [],
                  [
                   [7.7],
                   [8.8, 9.9]
                  ]])

    At `axis=0`, this operation pads the whole array, adding None at the
    outermost level:

        >>> ak.to_list(ak.pad_none(array, 5, axis=0))
        [[
          [1.1, 2.2, 3.3],
          [],
          [4.4, 5.5],
          [6.6]],
         [],
         [
          [7.7],
          [8.8, 9.9]
         ],
         None,
         None]

    At `axis=1`, this operation pads the first nested level:

        >>> ak.to_list(ak.pad_none(array, 3, axis=1))
        [[
          [1.1, 2.2, 3.3],
          [],
          [4.4, 5.5],
          [6.6]
         ],
         [
          None,
          None,
          None],
         [
          [7.7],
          [8.8, 9.9],
          None
         ]]

    And so on for higher values of `axis`:

        >>> ak.to_list(ak.pad_none(array, 2, axis=2))
        [[
          [1.1, 2.2, 3.3],
          [None, None],
          [4.4, 5.5],
          [6.6, None]
         ],
         [],
         [
          [7.7, None],
          [8.8, 9.9]
         ]]

    Note that the `clip` parameter not only determines whether the lengths are
    at least `target` or exactly `target`, it also determines the type of the
    output:

       * `clip=True` returns regular lists (#ak.types.RegularType), and
       * `clip=False` returns in-principle variable lengths
         (#ak.types.ListType).

    The in-principle variable-length lists might, in fact, all have the same
    length, but the type difference is significant, for instance in
    broadcasting rules (see #ak.broadcast_arrays).

    The difference between

        >>> ak.pad_none(array, 2, axis=2)
        <Array [[[1.1, 2.2, 3.3], ... [8.8, 9.9]]] type='3 * var * var * ?float64'>

    and

        >>> ak.pad_none(array, 2, axis=2, clip=True)
        <Array [[[1.1, 2.2], [None, ... [8.8, 9.9]]] type='3 * var * 2 * ?float64'>

    is not just in the length of `[1.1, 2.2, 3.3]` vs `[1.1, 2.2]`, but also
    in the distinction between the following types.

        >>> ak.type(ak.pad_none(array, 2, axis=2))
        3 * var * var * ?float64
        >>> ak.type(ak.pad_none(array, 2, axis=2, clip=True))
        3 * var *   2 * ?float64
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if clip:
        out = layout.rpad_and_clip(target, axis)
    else:
        out = layout.rpad(target, axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def fill_none(array, value, highlevel=True):
    """
    Args:
        array: Data in which to replace None with a given value.
        value: Data with which to replace None.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Replaces missing values (None) with a given `value`.

    For example, in the following `array`,

        ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    The None values could be replaced with `0` by

        >>> ak.fill_none(array, 0)
        <Array [[1.1, 0, 2.2], [], [0, 3.3, 4.4]] type='3 * var * float64'>

    The replacement value doesn't strictly need the same type as the
    surrounding data. For example, the None values could also be replaced
    by a string.

        >>> ak.fill_none(array, "hi")
        <Array [[1.1, 'hi', 2.2], ... ['hi', 3.3, 4.4]] type='3 * var * union[float64, s...'>

    The list content now has a union type:

        >>> ak.type(ak.fill_none(array, "hi"))
        3 * var * union[float64, string]

    The values could be floating-point numbers or strings.
    """
    arraylayout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )
    nplike = awkward1.nplike.of(arraylayout)

    if isinstance(arraylayout, awkward1.partition.PartitionedArray):
        out = awkward1.partition.apply(
            lambda x: fill_none(x, value, highlevel=False), arraylayout
        )

    else:
        if isinstance(value, Iterable) and not (
            isinstance(value, (str, bytes))
            or (awkward1._util.py27 and isinstance(value, awkward1._util.unicode))
        ):
            valuelayout = awkward1.operations.convert.to_layout(
                value, allow_record=True, allow_other=False
            )
            if isinstance(valuelayout, awkward1.layout.Record):
                valuelayout = valuelayout.array[valuelayout.at : valuelayout.at + 1]
            elif len(valuelayout) == 0:
                offsets = awkward1.layout.Index64(
                    nplike.array([0, 0], dtype=np.int64)
                )
                valuelayout = awkward1.layout.ListOffsetArray64(offsets, valuelayout)
            else:
                valuelayout = awkward1.layout.RegularArray(
                    valuelayout, len(valuelayout)
                )
        else:
            valuelayout = awkward1.operations.convert.to_layout(
                [value], allow_record=True, allow_other=False
            )

        out = arraylayout.fillna(valuelayout)

    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def is_none(array, highlevel=True):
    """
    Args:
        array: Data to check for missing values (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array whose value is True where an element of `array` is None;
    False otherwise.
    """

    def apply(layout):
        nplike = awkward1.nplike.of(layout)

        if isinstance(layout, awkward1._util.virtualtypes):
            return apply(layout.array)

        elif isinstance(layout, awkward1._util.unknowntypes):
            return apply(awkward1.layout.NumpyArray(nplike.array([])))

        elif isinstance(layout, awkward1._util.indexedtypes):
            return apply(layout.project())

        elif isinstance(layout, awkward1._util.uniontypes):
            contents = [apply(layout.project(i)) for i in range(layout.numcontents)]
            out = nplike.empty(len(layout), dtype=np.bool_)
            tags = nplike.asarray(layout.tags)
            for tag, content in enumerate(contents):
                out[tags == tag] = content
            return out

        elif isinstance(layout, awkward1._util.optiontypes):
            return nplike.asarray(layout.bytemask()).view(np.bool_)

        else:
            return nplike.zeros(len(layout), dtype=np.bool_)

    layout = awkward1.operations.convert.to_layout(array, allow_record=False)

    if isinstance(layout, awkward1.partition.PartitionedArray):
        out = awkward1.partition.apply(
            lambda x: awkward1.layout.NumpyArray(apply(x)), layout
        )
    else:
        out = awkward1.layout.NumpyArray(apply(layout))

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


def singletons(array, highlevel=True):
    """
    Args:
        array: Data to wrap in lists of length 1 if present and length 0
            if missing (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns a singleton list (length 1) wrapping each non-missing value and
    an empty list (length 0) in place of each missing value.

    For example,

        >>> array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
        >>> print(ak.singletons(array))
        [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]

    See #ak.firsts to invert this function.
    """

    def getfunction(layout, depth):
        nplike = awkward1.nplike.of(layout)

        if isinstance(layout, awkward1._util.optiontypes):
            nulls = nplike.asarray(layout.bytemask()).view(np.bool_)
            offsets = nplike.ones(len(layout) + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:][nulls] = 0
            nplike.cumsum(offsets, out=offsets)
            return lambda: awkward1.layout.ListOffsetArray64(
                awkward1.layout.Index64(offsets), layout.project()
            )
        else:
            return None

    layout = awkward1.operations.convert.to_layout(array)
    out = awkward1._util.recursively_apply(layout, getfunction)

    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def firsts(array, axis=1, highlevel=True):
    """
    Args:
        array: Data from which to select the first elements from nested lists.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Selects the first element of each non-empty list and inserts None for each
    empty list.

    For example,

        >>> array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])
        >>> print(ak.firsts(array))
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]

    See #ak.singletons to invert this function.
    """
    if axis <= 0:
        raise NotImplementedError(
            "ak.firsts with axis={0}".format(axis)
            + awkward1._util.exception_suffix(__file__)
        )
    toslice = (slice(None, None, None),) * axis + (0,)
    out = awkward1.mask(array, awkward1.num(array, axis=axis) > 0, highlevel=False)[
        toslice
    ]
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


def cartesian(
    arrays, axis=1, nested=None, parameters=None, with_name=None, highlevel=True
):
    """
    Args:
        arrays (dict or iterable of arrays): Arrays on which to compute the
            Cartesian product.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        nested (None, True, False, or iterable of str or int): If None or
            False, all combinations of elements from the `arrays` are
            produced at the same level of nesting; if True, they are grouped
            in nested lists by combinations that share a common item from
            each of the `arrays`; if an iterable of str or int, group common
            items for a chosen set of keys from the `array` dict or integer
            slots of the `array` iterable.
        parameters (None or dict): Parameters for the new
            #ak.layout.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.layout.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Computes a Cartesian product (i.e. cross product) of data from a set of
    `arrays`. This operation creates records (if `arrays` is a dict) or tuples
    (if `arrays` is another kind of iterable) that hold the combinations
    of elements, and it can introduce new levels of nesting.

    As a simple example with `axis=0`, the Cartesian product of

        >>> one = ak.Array([1, 2, 3])
        >>> two = ak.Array(["a", "b"])

    is

        >>> ak.to_list(ak.cartesian([one, two], axis=0))
        [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]

    With nesting, a new level of nested lists is created to group combinations
    that share the same element from `one` into the same list.

        >>> ak.to_list(ak.cartesian([one, two], axis=0, nested=True))
        [[(1, 'a'), (1, 'b')], [(2, 'a'), (2, 'b')], [(3, 'a'), (3, 'b')]]

    The primary purpose of this function, however, is to compute a different
    Cartesian product for each element of an array: in other words, `axis=1`.
    The following arrays each have four elements.

        >>> one = ak.Array([[1, 2, 3], [], [4, 5], [6]])
        >>> two = ak.Array([["a", "b"], ["c"], ["d"], ["e", "f"]])

    The default `axis=1` produces 6 pairs from the Cartesian product of
    `[1, 2, 3]` and `["a", "b"]`, 0 pairs from `[]` and `["c"]`, 1 pair from
    `[4, 5]` and `["d"]`, and 1 pair from `[6]` and `["e", "f"]`.

        >>> ak.to_list(ak.cartesian([one, two]))
        [[(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')],
         [],
         [(4, 'd'), (5, 'd')],
         [(6, 'e'), (6, 'f')]]

    The nesting depth is the same as the original arrays; with `nested=True`,
    the nesting depth is increased by 1 and tuples are grouped by their
    first element.

        >>> ak.to_list(ak.cartesian([one, two], nested=True))
        [[[(1, 'a'), (1, 'b')], [(2, 'a'), (2, 'b')], [(3, 'a'), (3, 'b')]],
         [],
         [[(4, 'd')], [(5, 'd')]],
         [[(6, 'e'), (6, 'f')]]]

    These tuples are #ak.layout.RecordArray nodes with unnamed fields. To
    name the fields, we can pass `one` and `two` in a dict, rather than a list.

        >>> ak.to_list(ak.cartesian({"x": one, "y": two}))
        [
         [{'x': 1, 'y': 'a'},
          {'x': 1, 'y': 'b'},
          {'x': 2, 'y': 'a'},
          {'x': 2, 'y': 'b'},
          {'x': 3, 'y': 'a'},
          {'x': 3, 'y': 'b'}],
         [],
         [{'x': 4, 'y': 'd'},
          {'x': 5, 'y': 'd'}],
         [{'x': 6, 'y': 'e'},
          {'x': 6, 'y': 'f'}]
        ]

    With more than two elements in the Cartesian product, `nested` can specify
    which are grouped and which are not. For example,

        >>> one = ak.Array([1, 2, 3, 4])
        >>> two = ak.Array([1.1, 2.2, 3.3])
        >>> three = ak.Array(["a", "b"])

    can be left entirely ungrouped:

        >>> ak.to_list(ak.cartesian([one, two, three], axis=0))
        [
         (1, 1.1, 'a'),
         (1, 1.1, 'b'),
         (1, 2.2, 'a'),
         (1, 2.2, 'b'),
         (1, 3.3, 'a'),
         (1, 3.3, 'b'),
         (2, 1.1, 'a'),
         (2, 1.1, 'b'),
         (2, 2.2, 'a'),
         (2, 2.2, 'b'),
         (2, 3.3, 'a'),
         (2, 3.3, 'b'),
         (3, 1.1, 'a'),
         (3, 1.1, 'b'),
         (3, 2.2, 'a'),
         (3, 2.2, 'b'),
         (3, 3.3, 'a'),
         (3, 3.3, 'b'),
         (4, 1.1, 'a'),
         (4, 1.1, 'b'),
         (4, 2.2, 'a'),
         (4, 2.2, 'b'),
         (4, 3.3, 'a'),
         (4, 3.3, 'b')
        ]

    can be grouped by `one` (adding 1 more dimension):

        >>> ak.to_list(ak.cartesian([one, two, three], axis=0, nested=[0]))
        [
         [(1, 1.1, 'a'), (1, 1.1, 'b'), (1, 2.2, 'a')],
         [(1, 2.2, 'b'), (1, 3.3, 'a'), (1, 3.3, 'b')],
         [(2, 1.1, 'a'), (2, 1.1, 'b'), (2, 2.2, 'a')],
         [(2, 2.2, 'b'), (2, 3.3, 'a'), (2, 3.3, 'b')],
         [(3, 1.1, 'a'), (3, 1.1, 'b'), (3, 2.2, 'a')],
         [(3, 2.2, 'b'), (3, 3.3, 'a'), (3, 3.3, 'b')],
         [(4, 1.1, 'a'), (4, 1.1, 'b'), (4, 2.2, 'a')],
         [(4, 2.2, 'b'), (4, 3.3, 'a'), (4, 3.3, 'b')]
        ]

    can be grouped by `one` and `two` (adding 2 more dimensions):

        >>> ak.to_list(ak.cartesian([one, two, three], axis=0, nested=[0, 1]))
        [
         [
          [(1, 1.1, 'a'), (1, 1.1, 'b')],
          [(1, 2.2, 'a'), (1, 2.2, 'b')],
          [(1, 3.3, 'a'), (1, 3.3, 'b')]
         ],
         [
          [(2, 1.1, 'a'), (2, 1.1, 'b')],
          [(2, 2.2, 'a'), (2, 2.2, 'b')],
          [(2, 3.3, 'a'), (2, 3.3, 'b')]
         ],
         [
          [(3, 1.1, 'a'), (3, 1.1, 'b')],
          [(3, 2.2, 'a'), (3, 2.2, 'b')],
          [(3, 3.3, 'a'), (3, 3.3, 'b')]],
         [
          [(4, 1.1, 'a'), (4, 1.1, 'b')],
          [(4, 2.2, 'a'), (4, 2.2, 'b')],
          [(4, 3.3, 'a'), (4, 3.3, 'b')]]
        ]

    or grouped by unique `one`-`two` pairs (adding 1 more dimension):

        >>> ak.to_list(ak.cartesian([one, two, three], axis=0, nested=[1]))
        [
         [(1, 1.1, 'a'), (1, 1.1, 'b')],
         [(1, 2.2, 'a'), (1, 2.2, 'b')],
         [(1, 3.3, 'a'), (1, 3.3, 'b')],
         [(2, 1.1, 'a'), (2, 1.1, 'b')],
         [(2, 2.2, 'a'), (2, 2.2, 'b')],
         [(2, 3.3, 'a'), (2, 3.3, 'b')],
         [(3, 1.1, 'a'), (3, 1.1, 'b')],
         [(3, 2.2, 'a'), (3, 2.2, 'b')],
         [(3, 3.3, 'a'), (3, 3.3, 'b')],
         [(4, 1.1, 'a'), (4, 1.1, 'b')],
         [(4, 2.2, 'a'), (4, 2.2, 'b')],
         [(4, 3.3, 'a'), (4, 3.3, 'b')]
        ]

    The order of the output is fixed: it is always lexicographical in the
    order that the `arrays` are written. (Before Python 3.6, the order of
    keys in a dict were not guaranteed, so the dict interface is not
    recommended for these versions of Python.) Thus, it is not possible to
    group by `three` in the example above.

    To emulate an SQL or Pandas "group by" operation, put the keys that you
    wish to group by *first* and use `nested=[0]` or `nested=[n]` to group by
    unique n-tuples. If necessary, record keys can later be reordered with a
    list of strings in #ak.Array.__getitem__.

    To get list index positions in the tuples/records, rather than data from
    the original `arrays`, use #ak.argcartesian instead of #ak.cartesian. The
    #ak.argcartesian form can be particularly useful as nested indexing in
    #ak.Array.__getitem__.
    """
    behavior = awkward1._util.behaviorof(*arrays)
    nplike = awkward1.nplike.of(*arrays)

    is_partitioned = False
    if isinstance(arrays, dict):
        new_arrays = {}
        for n, x in arrays.items():
            new_arrays[n] = awkward1.operations.convert.to_layout(
                x, allow_record=False, allow_other=False
            )
            if isinstance(new_arrays[n], awkward1.partition.PartitionedArray):
                is_partitioned = True
    else:
        new_arrays = []
        for x in arrays:
            new_arrays.append(
                awkward1.operations.convert.to_layout(
                    x, allow_record=False, allow_other=False
                )
            )
            if isinstance(new_arrays[-1], awkward1.partition.PartitionedArray):
                is_partitioned = True

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    if axis < 0:
        raise ValueError(
            "the 'axis' of cartesian must be non-negative"
            + awkward1._util.exception_suffix(__file__)
        )

    elif axis == 0:
        if nested is None or nested is False:
            nested = []

        if isinstance(new_arrays, dict):
            if nested is True:
                nested = list(new_arrays.keys())  # last key is ignored below
            if any(not (isinstance(n, str) and n in new_arrays) for x in nested):
                raise ValueError(
                    "the 'nested' parameter of cartesian must be dict keys "
                    "for a dict of arrays"
                    + awkward1._util.exception_suffix(__file__)
                )
            recordlookup = []
            layouts = []
            tonested = []
            for i, (n, x) in enumerate(new_arrays.items()):
                recordlookup.append(n)
                layouts.append(x)
                if n in nested:
                    tonested.append(i)
            nested = tonested

        else:
            if nested is True:
                nested = list(range(len(new_arrays) - 1))
            if any(
                not (isinstance(x, int) and 0 <= x < len(new_arrays) - 1)
                for x in nested
            ):
                raise ValueError(
                    "the 'nested' prarmeter of cartesian must be integers in "
                    "[0, len(arrays) - 1) for an iterable of arrays"
                    + awkward1._util.exception_suffix(__file__)
                )
            recordlookup = None
            layouts = []
            for x in new_arrays:
                layouts.append(x)

        layouts = [
            x.toContent() if isinstance(x, awkward1.partition.PartitionedArray) else x
            for x in layouts
        ]

        indexes = [
            awkward1.layout.Index64(x.reshape(-1))
            for x in nplike.meshgrid(
                *[nplike.arange(len(x), dtype=np.int64) for x in layouts],
                indexing="ij"
            )
        ]
        outs = [
            awkward1.layout.IndexedArray64(x, y)
            for x, y in __builtins__["zip"](indexes, layouts)
        ]

        result = awkward1.layout.RecordArray(outs, recordlookup, parameters=parameters)
        for i in range(len(new_arrays) - 1, -1, -1):
            if i in nested:
                result = awkward1.layout.RegularArray(result, len(layouts[i + 1]))

    elif is_partitioned:
        sample = None
        if isinstance(new_arrays, dict):
            for x in new_arrays.values():
                if isinstance(x, awkward1.partition.PartitionedArray):
                    sample = x
                    break
        else:
            for x in new_arrays:
                if isinstance(x, awkward1.partition.PartitionedArray):
                    sample = x
                    break

        partition_arrays = awkward1.partition.partition_as(sample, new_arrays)

        output = []
        for part_arrays in awkward1.partition.iterate(
            sample.numpartitions, partition_arrays
        ):
            output.append(
                cartesian(
                    part_arrays,
                    axis=axis,
                    nested=nested,
                    parameters=parameters,
                    with_name=None,  # already set: see above
                    highlevel=False,
                )
            )

        result = awkward1.partition.IrregularlyPartitionedArray(output)

    else:

        def newaxis(layout, i):
            if i == 0:
                return layout
            else:
                return awkward1.layout.RegularArray(newaxis(layout, i - 1), 1)

        def getfunction1(layout, depth, i):
            if depth == 2:
                return lambda: newaxis(layout, i)
            else:
                return None

        def getfunction2(layout, depth, i):
            if depth == axis:
                inside = len(new_arrays) - i - 1
                outside = i
                return lambda: newaxis(
                    awkward1._util.recursively_apply(
                        layout, getfunction1, args=(inside,)
                    ),
                    outside,
                )
            else:
                return None

        def apply(x, i):
            return awkward1._util.recursively_apply(
                awkward1.operations.convert.to_layout(
                    x, allow_record=False, allow_other=False
                ),
                getfunction2,
                args=(i,),
            )

        toflatten = []
        if nested is None or nested is False:
            nested = []

        if isinstance(new_arrays, dict):
            if nested is True:
                nested = list(new_arrays.keys())  # last key is ignored below
            if any(not (isinstance(n, str) and n in new_arrays) for x in nested):
                raise ValueError(
                    "the 'nested' parameter of cartesian must be dict keys "
                    "for a dict of arrays"
                    + awkward1._util.exception_suffix(__file__)
                )
            recordlookup = []
            layouts = []
            for i, (n, x) in enumerate(new_arrays.items()):
                recordlookup.append(n)
                layouts.append(apply(x, i))
                if i < len(new_arrays) - 1 and n not in nested:
                    toflatten.append(axis + i + 1)

        else:
            if nested is True:
                nested = list(range(len(new_arrays) - 1))
            if any(
                not (isinstance(x, int) and 0 <= x < len(new_arrays) - 1)
                for x in nested
            ):
                raise ValueError(
                    "the 'nested' parameter of cartesian must be integers in "
                    "[0, len(arrays) - 1) for an iterable of arrays"
                    + awkward1._util.exception_suffix(__file__)
                )
            recordlookup = None
            layouts = []
            for i, x in enumerate(new_arrays):
                layouts.append(apply(x, i))
                if i < len(new_arrays) - 1 and i not in nested:
                    toflatten.append(axis + i + 1)

        def getfunction3(inputs, depth):
            if depth == axis + len(new_arrays):
                return lambda: (
                    awkward1.layout.RecordArray(
                        inputs, recordlookup, parameters=parameters
                    ),
                )
            else:
                return None

        out = awkward1._util.broadcast_and_apply(layouts, getfunction3, behavior)
        assert isinstance(out, tuple) and len(out) == 1
        result = out[0]

        while len(toflatten) != 0:
            axis = toflatten.pop()
            result = flatten(result, axis=axis, highlevel=False)

    if highlevel:
        return awkward1._util.wrap(result, behavior)
    else:
        return result


def argcartesian(
    arrays, axis=1, nested=None, parameters=None, with_name=None, highlevel=True
):
    """
    Args:
        arrays (dict or iterable of arrays): Arrays on which to compute the
            Cartesian product.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        nested (None, True, False, or iterable of str or int): If None or
            False, all combinations of elements from the `arrays` are
            produced at the same level of nesting; if True, they are grouped
            in nested lists by combinations that share a common item from
            each of the `arrays`; if an iterable of str or int, group common
            items for a chosen set of keys from the `array` dict or slots
            of the `array` iterable.
        parameters (None or dict): Parameters for the new
            #ak.layout.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.layout.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Computes a Cartesian product (i.e. cross product) of data from a set of
    `arrays`, like #ak.cartesian, but returning integer indexes for
    #ak.Array.__getitem__.

    For example, the Cartesian product of

        >>> one = ak.Array([1.1, 2.2, 3.3])
        >>> two = ak.Array(["a", "b"])

    is

        >>> ak.to_list(ak.cartesian([one, two], axis=0))
        [(1.1, 'a'), (1.1, 'b'), (2.2, 'a'), (2.2, 'b'), (3.3, 'a'), (3.3, 'b')]

    But with argcartesian, only the indexes are returned.

        >>> ak.to_list(ak.argcartesian([one, two], axis=0))
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    These are the indexes that can select the items that go into the actual
    Cartesian product.

        >>> one_index, two_index = ak.unzip(ak.argcartesian([one, two], axis=0))
        >>> one[one_index]
        <Array [1.1, 1.1, 2.2, 2.2, 3.3, 3.3] type='6 * float64'>
        >>> two[two_index]
        <Array ['a', 'b', 'a', 'b', 'a', 'b'] type='6 * string'>

    All of the parameters for #ak.cartesian apply equally to #ak.argcartesian,
    so see the #ak.cartesian documentation for a more complete description.
    """
    if axis < 0:
        raise ValueError(
            "the 'axis' of argcartesian must be non-negative"
            + awkward1._util.exception_suffix(__file__)
        )

    else:
        if isinstance(arrays, dict):
            layouts = dict(
                (
                    n,
                    awkward1.operations.convert.to_layout(
                        x, allow_record=False, allow_other=False
                    ).localindex(axis),
                )
                for n, x in arrays.items()
            )
        else:
            layouts = [
                awkward1.operations.convert.to_layout(
                    x, allow_record=False, allow_other=False
                ).localindex(axis)
                for x in arrays
            ]

        if with_name is not None:
            if parameters is None:
                parameters = {}
            else:
                parameters = dict(parameters)
            parameters["__record__"] = with_name

        result = cartesian(
            layouts, axis=axis, nested=nested, parameters=parameters, highlevel=False
        )

        if highlevel:
            return awkward1._util.wrap(result, awkward1._util.behaviorof(*arrays))
        else:
            return result


def combinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
):
    """
    Args:
        array: Array from which to choose `n` items without replacement.
        n (int): The number of items to choose in each list: `2` chooses
            unique pairs, `3` chooses unique triples, etc.
        replacement (bool): If True, combinations that include the same
            item more than once are allowed; otherwise each item in a
            combinations is strictly unique.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        fields (None or list of str): If None, the pairs/triples/etc. are
            tuples with unnamed fields; otherwise, these `fields` name the
            fields. The number of `fields` must be equal to `n`.
        parameters (None or dict): Parameters for the new
            #ak.layout.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.layout.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Computes a Cartesian product (i.e. cross product) of `array` with itself
    that is restricted to combinations sampled without replacement. If the
    normal Cartesian product is thought of as an `n` dimensional tensor, these
    represent the "upper triangle" of sets without repetition. If
    `replacement=True`, the diagonal of this "upper triangle" is included.

    As a simple example with `axis=0`, consider the following `array`

        ak.Array(["a", "b", "c", "d", "e"])

    The combinations choose `2` are:

        >>> ak.to_list(ak.combinations(array, 2, axis=0))
        [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
                     ('b', 'c'), ('b', 'd'), ('b', 'e'),
                                 ('c', 'd'), ('c', 'e'),
                                             ('d', 'e')]

    Including the diagonal allows pairs like `('a', 'a')`.

        >>> ak.to_list(ak.combinations(array, 2, axis=0, replacement=True))
        [('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'),
                     ('b', 'b'), ('b', 'c'), ('b', 'd'), ('b', 'e'),
                                 ('c', 'c'), ('c', 'd'), ('c', 'e'),
                                             ('d', 'd'), ('d', 'e'),
                                                         ('e', 'e')]

    The combinations choose `3` can't be easily arranged as a triangle
    in two dimensions.

        >>> ak.to_list(ak.combinations(array, 3, axis=0))
        [('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'e'), ('a', 'c', 'd'), ('a', 'c', 'e'),
         ('a', 'd', 'e'), ('b', 'c', 'd'), ('b', 'c', 'e'), ('b', 'd', 'e'), ('c', 'd', 'e')]

    Including the (three-dimensional) diagonal allows triples like
    `('a', 'a', 'a')`, but also `('a', 'a', 'b')`, `('a', 'b', 'b')`, etc.,
    but not `('a', 'b', 'a')`. All combinations are in the same order as
    the original array.

        >>> ak.to_list(ak.combinations(array, 3, axis=0, replacement=True))
        [('a', 'a', 'a'), ('a', 'a', 'b'), ('a', 'a', 'c'), ('a', 'a', 'd'), ('a', 'a', 'e'),
         ('a', 'b', 'b'), ('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'e'), ('a', 'c', 'c'),
         ('a', 'c', 'd'), ('a', 'c', 'e'), ('a', 'd', 'd'), ('a', 'd', 'e'), ('a', 'e', 'e'),
         ('b', 'b', 'b'), ('b', 'b', 'c'), ('b', 'b', 'd'), ('b', 'b', 'e'), ('b', 'c', 'c'),
         ('b', 'c', 'd'), ('b', 'c', 'e'), ('b', 'd', 'd'), ('b', 'd', 'e'), ('b', 'e', 'e'),
         ('c', 'c', 'c'), ('c', 'c', 'd'), ('c', 'c', 'e'), ('c', 'd', 'd'), ('c', 'd', 'e'),
         ('c', 'e', 'e'), ('d', 'd', 'd'), ('d', 'd', 'e'), ('d', 'e', 'e'), ('e', 'e', 'e')]

    The primary purpose of this function, however, is to compute a different
    set of combinations for each element of an array: in other words, `axis=1`.
    The following `array` has a different number of items in each element.

        ak.Array([[1, 2, 3, 4], [], [5], [6, 7, 8]])

    There are 6 ways to choose pairs from 4 elements, 0 ways to choose pairs
    from 0 elements, 0 ways to choose pairs from 1 element, and 3 ways to
    choose pairs from 3 elements.

        >>> ak.to_list(ak.combinations(array, 2))
        [
         [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
         [],
         [],
         [(6, 7), (6, 8), (7, 8)]
        ]

    Note, however, that the combinatorics isn't determined by equality of
    the data themselves, but by their placement in the array. For example,
    even if all elements of an array are equal, the output has the same
    structure.

        >>> same = ak.Array([[7, 7, 7, 7], [], [7], [7, 7, 7]])
        >>> ak.to_list(ak.combinations(same, 2))
        [
         [(7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7)],
         [],
         [],
         [(7, 7), (7, 7), (7, 7)]
        ]

    To get records instead of tuples, pass a set of field names to `fields`.

        >>> ak.to_list(ak.combinations(array, 2, fields=["x", "y"]))
        [
         [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4},
                            {'x': 2, 'y': 3}, {'x': 2, 'y': 4},
                                              {'x': 3, 'y': 4}],
         [],
         [],
         [{'x': 6, 'y': 7}, {'x': 6, 'y': 8},
                            {'x': 7, 'y': 8}]]

    This operation can be constructed from #ak.argcartesian and other
    primitives:

        >>> left, right = ak.unzip(ak.argcartesian([array, array]))
        >>> keep = left < right
        >>> result = ak.zip([array[left][keep], array[right][keep]])
        >>> ak.to_list(result)
        [
         [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
         [],
         [],
         [(6, 7), (6, 8), (7, 8)]]

    but it is frequently needed for data analysis, and the logic of which
    indexes to `keep` (above) gets increasingly complicated for large `n`.

    To get list index positions in the tuples/records, rather than data from
    the original `array`, use #ak.argcombinations instead of #ak.combinations.
    The #ak.argcombinations form can be particularly useful as nested indexing
    in #ak.Array.__getitem__.
    """
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters)
    if with_name is not None:
        parameters["__record__"] = with_name

    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    out = layout.combinations(
        n, replacement=replacement, keys=fields, parameters=parameters, axis=axis
    )
    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


def argcombinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
):
    """
    Args:
        array: Array from which to choose `n` items without replacement.
        n (int): The number of items to choose from each list: `2` chooses
            unique pairs, `3` chooses unique triples, etc.
        replacement (bool): If True, combinations that include the same
            item more than once are allowed; otherwise each item in a
            combinations is strictly unique.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        fields (None or list of str): If None, the pairs/triples/etc. are
            tuples with unnamed fields; otherwise, these `fields` name the
            fields. The number of `fields` must be equal to `n`.
        parameters (None or dict): Parameters for the new
            #ak.layout.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.layout.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Computes a Cartesian product (i.e. cross product) of `array` with itself
    that is restricted to combinations sampled without replacement,
    like #ak.combinations, but returning integer indexes for
    #ak.Array.__getitem__.

    The motivation and uses of this function are similar to those of
    #ak.argcartesian. See #ak.combinations and #ak.argcartesian for a more
    complete description.
    """
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters)
    if with_name is not None:
        parameters["__record__"] = with_name

    if axis < 0:
        raise ValueError(
            "the 'axis' for argcombinations must be non-negative"
            + awkward1._util.exception_suffix(__file__)
        )
    else:
        layout = awkward1.operations.convert.to_layout(
            array, allow_record=False, allow_other=False
        ).localindex(axis)
        out = layout.combinations(
            n, replacement=replacement, keys=fields, parameters=parameters, axis=axis
        )
        if highlevel:
            return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
        else:
            return out


def partitions(array):
    """
    Args:
        array: A possibly-partitioned array.

    Returns a list of partition lengths if the array is a PartitionedArray;
    returns None otherwise.

    Partitioning is an internal aspect of an array: it should behave
    identically to a non-partitioned array, but possibly with different
    performance characteristics.

    Arrays can only be partitioned in the first dimension; it is intended
    for performing calculations in memory-sized chunks.
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if isinstance(layout, awkward1.partition.PartitionedArray):
        return layout.lengths
    else:
        return None


def partitioned(generate, numpartitions, highlevel=True, behavior=None):
    """
    Args:
        generate (int -> array): A function that generates an array partition
            given a partition number.
        numpartitions (int): Number of partitions to generate.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content or #ak.partition.PartitionedArray
            subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Returns a partitioned array, produced by calling a function for each
    partition.

    Partitioning is an internal aspect of an array: it should behave
    identically to a non-partitioned array, but possibly with different
    performance characteristics.

    Arrays can only be partitioned in the first dimension; it is intended
    for performing calculations in memory-sized chunks.
    """
    total_length = 0
    partitions = []
    stops = []
    for partitionid in range(numpartitions):
        layout = awkward1.operations.convert.to_layout(
            generate(partitionid), allow_record=False, allow_other=False
        )
        total_length += len(layout)
        partitions.append(layout)
        stops.append(total_length)

    out = awkward1.partition.IrregularlyPartitionedArray(partitions, stops)
    if highlevel:
        return awkward1._util.wrap(out, behavior=behavior)
    else:
        return out


def repartition(array, lengths, highlevel=True):
    """
    Args:
        array: A possibly-partitioned array.
        lengths (None, int, or iterable of int): If None, concatenate the
            pieces of a partitioned array into a non-partitioned array.
            If an integer, split or repartition into partitions of the
            given number of entries (except the last, if the length of the
            array doesn't fit an integer number of equal-sized chunks).
            If an iterable of integers, split or repartition into the given
            sequence of lengths.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content or #ak.partition.PartitionedArray
            subclass.

    Returns a possibly-partitioned array: unpartitioned if `lengths` is None;
    partitioned otherwise.

    Partitioning is an internal aspect of an array: it should behave
    identically to a non-partitioned array, but possibly with different
    performance characteristics.

    Arrays can only be partitioned in the first dimension; it is intended
    for performing calculations in memory-sized chunks.
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    if lengths is None:
        if isinstance(layout, awkward1.partition.PartitionedArray):
            out = layout.toContent()
        else:
            out = layout

    else:
        if isinstance(lengths, (int, numbers.Integral, np.integer)):
            if lengths < 1:
                raise ValueError(
                    "lengths must be at least 1 (and probably considerably more)"
                    + awkward1._util.exception_suffix(__file__)
                )

            howmany = len(layout) // lengths
            remainder = len(layout) - howmany * lengths
            if remainder == 0:
                lengths = [lengths] * howmany
            else:
                lengths = [lengths] * howmany + [remainder]

        total_length = 0
        stops = []
        for x in lengths:
            total_length += x
            stops.append(total_length)

        if total_length != len(layout):
            raise ValueError(
                "cannot repartition array of length {0} into "
                "these lengths".format(len(layout))
                + awkward1._util.exception_suffix(__file__)
            )

        if isinstance(layout, awkward1.partition.PartitionedArray):
            out = layout.repartition(stops)
        else:
            out = awkward1.partition.IrregularlyPartitionedArray.toPartitioned(
                layout, stops
            )

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out


def virtual(
    generate,
    args=(),
    kwargs={},
    form=None,
    length=None,
    cache=None,
    cache_key=None,
    parameters=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        generate (callable): Function that makes an array from `args` and
            `kwargs`.
        args (tuple): Positional arguments to pass to `generate`.
        kwargs (dict): Keyword arguments to pass to `generate`.
        form (None, Form, or JSON): If None, the layout of the generated array
            is unknown until it is generated, which might require it to be
            generated earlier than intended; if a Form, use this Form to
            predict the layout and verify that the generated array complies;
            if a JSON string, convert the JSON into a Form and use it.
        length (None or int): If None or negative, the length of the generated
            array is unknown until it is generated, which might require it to
            be generated earlier than intended; if a non-negative int, use this
            to predict the length and verify that the generated array complies.
        cache (None or MutableMapping): If None, arrays are generated every
            time they are needed; otherwise, generated arrays are stored in the
            mapping with `__setitem__`, retrieved with `__getitem__`, and only
            re-generated if `__getitem__` raises a `KeyError`. This mapping may
            evict elements according to any caching algorithm (LRU, LFR, RR,
            TTL, etc.).
        cache_key (None or str): If None, a unique string is generated for this
            virtual array for use with the `cache` (unique per Python process);
            otherwise, the explicitly provided key is used (which ought to
            ensure global uniquness for the scope in which these arrays are
            used).
        parameters (None or dict): Parameters for the new
            #ak.layout.VirtualArray node that is created by this operation.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.

    Creates a virtual array, an array that is created on demand.

    For example:

        >>> len(array)
        3
        >>> awkward1.type(array)
        3 * var * float64
        >>> array[2]
        generating
        <Array [4.4, 5.5] type='2 * float64'>
    """
    if isinstance(form, str) and form in (
        "float64",
        "float32",
        "int64",
        "uint64",
        "int32",
        "uint32",
        "int16",
        "uint16",
        "int8",
        "uint8",
        "bool",
    ):
        form = awkward1.forms.Form.fromjson('"' + form + '"')

    elif isinstance(form, (str, bytes)) or (
        awkward1._util.py27 and isinstance(form, awkward1._util.unicode)
    ):
        form = awkward1.forms.Form.fromjson(form)

    elif form is not None and not isinstance(form, awkward1.forms.Form):
        form = awkward1.forms.Form.fromjson(json.dumps(form))

    gen = awkward1.layout.ArrayGenerator(
        generate, args, kwargs, form=form, length=length
    )
    if cache is not None and not isinstance(cache, awkward1.layout.ArrayCache):
        maybe_wrapped = awkward1._util.MappingProxy.maybe_wrap(cache)
        cache = awkward1.layout.ArrayCache(maybe_wrapped)

    out = awkward1.layout.VirtualArray(
        gen, cache, cache_key=cache_key, parameters=parameters
    )

    if highlevel:
        return awkward1._util.wrap(out, behavior=behavior)
    else:
        return out


def with_cache(array, cache, highlevel=True):
    """
    Args:
        array: Data to search for nested virtual arrays.
        cache (None or MutableMapping): If None, arrays are generated every
            time they are needed; otherwise, generated arrays are stored in the
            mapping with `__setitem__`, retrieved with `__getitem__`, and only
            re-generated if `__getitem__` raises a `KeyError`. This mapping may
            evict elements according to any caching algorithm (LRU, LFR, RR,
            TTL, etc.).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Remove caches from all virtual arrays nested within `array` if `cache` is
    None; adds a cache otherwise.

    For example:

        >>> cache1 = {}
        >>> one = ak.virtual(lambda: [[1.1, 2.2, 3.3], [], [4.4, 5.5]], cache=cache1, length=3)
        >>> two = ak.virtual(lambda: [100, 200, 300], cache=cache1, length=3)
        >>> array1 = ak.zip({"x": one, "y": two}, depth_limit=1)
        >>> len(cache1)
        0

    creates an array of records with virtual fields that would fill `cache1`.

    We can then switch every instance of `cache1` in `array` with `cache2`:

        >>> cache2 = {}
        >>> array2 = ak.with_cache(array1, cache2)
        >>> array2["x"]
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
        >>>
        >>> len(cache1), len(cache2)
        (0, 1)

    Viewing the `array2["x"]` filled `cache2` and not `cache1`.

    See #ak.virtual.
    """
    if cache is not None and not isinstance(cache, awkward1.layout.ArrayCache):
        maybe_wrapped = awkward1._util.MappingProxy.maybe_wrap(cache)
        cache = awkward1.layout.ArrayCache(maybe_wrapped)

    def getfunction(layout, depth):
        if isinstance(layout, awkward1.layout.VirtualArray):
            if cache is None:
                newcache = layout.cache
            elif layout.cache is None:
                newcache = cache
            return lambda: awkward1.layout.VirtualArray(
                layout.generator,
                newcache,
                layout.cache_key,
                layout.identities,
                layout.parameters,
            )
        else:
            return None

    out = awkward1._util.recursively_apply(
        awkward1.operations.convert.to_layout(array), getfunction
    )
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


@awkward1._connect._numpy.implements("size")
def size(array, axis=None):
    """
    Args:
        array: Rectilinear array whose `shape` needs to be known.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.

    Returns an int or a list of ints, one for each regular dimension.

    Implements NumPy's
    [size](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html)
    function in a way that accepts #ak.Array as the `array`.

    If the `array` is not rectilinear (i.e. if #np.to_numpy would raise an
    error), then this function raise an error.
    """
    if axis is not None and axis < 0:
        raise NotImplementedError(
            "ak.size with axis < 0"
            + awkward1._util.exception_suffix(__file__)
        )

    def recurse(layout, axis, sizes):
        nplike = awkward1.nplike.of(layout)

        if isinstance(layout, awkward1._util.virtualtypes):
            recurse(layout.array, axis, sizes)
        elif isinstance(layout, awkward1._util.unknowntypes):
            pass
        elif isinstance(layout, awkward1._util.indexedtypes):
            recurse(layout.content, axis, sizes)
        elif isinstance(layout, awkward1._util.uniontypes):
            compare = None
            for x in layout.contents:
                inner = []
                recurse(x, axis, inner)
                if compare is None:
                    compare = inner
                elif compare != inner:
                    raise ValueError(
                        "ak.size is ambiguous due to union of different " "sizes"
                        + awkward1._util.exception_suffix(__file__)
                    )
            sizes.extend(compare)
        elif isinstance(layout, awkward1._util.optiontypes):
            return recurse(layout.content, axis, sizes)
        elif isinstance(layout, awkward1._util.listtypes):
            if isinstance(layout, awkward1.layout.RegularArray):
                sizes.append(layout.size)
            else:
                sizes.append(None)
            if axis is None:
                recurse(layout.content, axis, sizes)
            elif axis > 0:
                recurse(layout.content, axis - 1, sizes)
        elif isinstance(layout, awkward1._util.recordtypes):
            compare = None
            for x in layout.contents:
                inner = []
                recurse(x, axis, inner)
                if compare is None:
                    compare = inner
                elif compare != inner:
                    raise ValueError(
                        "ak.size is ambiguous due to record of different " "sizes"
                        + awkward1._util.exception_suffix(__file__)
                    )
            sizes.extend(compare)
        elif isinstance(layout, awkward1.layout.NumpyArray):
            if axis is None:
                sizes.extend(nplike.asarray(layout).shape[1:])
            else:
                sizes.extend(nplike.asarray(layout).shape[1 : axis + 2])
        else:
            raise AssertionError(
                "unrecognized Content type"
                + awkward1._util.exception_suffix(__file__)
            )

    layout = awkward1.operations.convert.to_layout(array, allow_record=False)
    if isinstance(layout, awkward1.partition.PartitionedArray):
        layout = layout.toContent()
    layout = awkward1.layout.RegularArray(layout, len(layout))

    sizes = []
    recurse(layout, axis, sizes)

    if axis is None:
        out = 1
        for size in sizes:
            if size is None:
                raise ValueError(
                    "ak.size is ambiguous due to variable-length arrays "
                    "(try ak.flatten to remove structure or ak.to_numpy "
                    "to force regularity, if possible)"
                    + awkward1._util.exception_suffix(__file__)
                )
            else:
                out *= size
        return out
    else:
        if sizes[-1] is None:
            raise ValueError(
                "ak.size is ambiguous due to variable-length arrays at "
                "axis {0} (try ak.flatten to remove structure or "
                "ak.to_numpy to force regularity, if possible)".format(axis)
                + awkward1._util.exception_suffix(__file__)
            )
        else:
            return sizes[-1]


@awkward1._connect._numpy.implements("atleast_1d")
def atleast_1d(*arrays):
    """
    Args:
        arrays: Rectilinear arrays to be converted to NumPy arrays of at
            least 1 dimension.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.

    Implements NumPy's
    [atleast_1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html)
    function in a way that accepts #ak.Array objects as the `arrays`.

    If the `arrays` are not all rectilinear (i.e. if #np.to_numpy would raise an
    error), then this function raise an error.

    Note: this function returns a NumPy array, not an Awkward Array.
    """
    nplike = awkward1.nplike.of(*arrays)
    return nplike.atleast_1d(*[awkward1.operations.convert.to_numpy(x) for x in arrays])

_dtype_to_string = {
    np.dtype(np.bool): "bool",
    np.dtype(np.bool_): "bool",
    np.dtype(np.int8): "int8",
    np.dtype(np.int16): "int16",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.uint64): "uint64",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    np.dtype(np.complex64): "complex64",
    np.dtype(np.complex128): "complex128",
    # np.dtype(np.datetime64): "datetime64",
    # np.dtype(np.timedelta64): "timedelta64",
}

if hasattr(np, "float16"):
    _dtype_to_string[np.dtype(np.float16)] = "float16"
if hasattr(np, "float128"):
    _dtype_to_string[np.dtype(np.float128)] = "float128"
if hasattr(np, "complex256"):
    _dtype_to_string[np.dtype(np.complex256)] = "complex256"

def values_astype(array, to, highlevel=True):
    """
    Args:
        array: Array whose numbers should be converted to a new numeric type.
        to (dtype or dtype specifier): Type to convert the numbers into.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Converts all numbers in the array to a new type, leaving the structure
    untouched.
    """
    to_dtype = np.dtype(to)
    to_str = _dtype_to_string.get(to_dtype)
    if to_str is None:
        raise ValueError(
            "cannot use {0} to cast the numeric type of an array".format(to_dtype)
            + awkward1._util.exception_suffix(__file__)
        )

    layout = awkward1.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    out = layout.numbers_to_type(to_str)

    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_") and x not in (
        "absolute_import",
        "numbers",
        "json",
        "Iterable",
        "MutableMapping",
        "np",
        "awkward1",
    )
]
