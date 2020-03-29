# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout
import awkward1._connect._numpy
import awkward1.operations.convert

def withfield(base, what, where=None, highlevel=True):
    """
    Args:
        base: Data containing records or tuples.
        what: Data to add as a new field.
        where (None or str): If None, the new field has no name (can be
            accessed as an integer slot number in a string); otherwise, the
            name of the new field.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new field attached. This function does not
    change the array in-place.

    See #ak.Array.__setitem__ and #ak.Record.__setitem__ for a variant that
    changes the high-level object in-place. (These methods internally use
    #ak.withfield, so performance is not a factor in choosing one over the
    other.)
    """
    base = awkward1.operations.convert.tolayout(base,
                                                allowrecord=True,
                                                allowother=False)
    if base.numfields < 0:
        raise ValueError("no tuples or records in array; cannot add a new "
                "field")

    what = awkward1.operations.convert.tolayout(what,
                                                allowrecord=True,
                                                allowother=True)
    
    keys = base.keys()
    if where in base.keys():
        keys.remove(where)
        base = base[keys]

    def getfunction(inputs, depth):
        base, what = inputs
        if isinstance(base, awkward1.layout.RecordArray):
            if not isinstance(what, awkward1.layout.Content):
                what = awkward1.layout.NumpyArray(
                    numpy.lib.stride_tricks.as_strided(
                      [what], shape=(len(base),), strides=(0,)))
            return lambda: (base.setitem_field(where, what),)
        else:
            return None

    behavior = awkward1._util.behaviorof(base, what)
    out = awkward1._util.broadcast_and_apply(
            [base, what], getfunction, behavior)
    assert isinstance(out, tuple) and len(out) == 1
    if highlevel:
        return awkward1._util.wrap(out[0], behavior=behavior)
    else:
        return out[0]

def withname(array, name, highlevel=True):
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
                parameters)
        else:
            return None
    out = awkward1._util.recursively_apply(
            awkward1.operations.convert.tolayout(array), getfunction)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def isna(array, highlevel=True):
    """
    Args:
        array: Data to check for missing values (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array whose value is True where an element of `array` is None;
    False otherwise.

    See also #ak.notna, the logical negation of #ak.isna.
    """
    def apply(layout):
        if isinstance(layout, awkward1._util.unknowntypes):
            return apply(awkward1.layout.NumpyArray(numpy.array([])))

        elif isinstance(layout, awkward1._util.indexedtypes):
            return apply(layout.project())

        elif isinstance(layout, awkward1._util.uniontypes):
            contents = [apply(layout.project(i))
                          for i in range(layout.numcontents)]
            out = numpy.empty(len(layout), dtype=numpy.bool_)
            tags = numpy.asarray(layout.tags)
            for tag, content in enumerate(contents):
                out[tags == tag] = content
            return out

        elif isinstance(layout, awkward1._util.optiontypes):
            index = numpy.asarray(layout.index)
            return (index < 0)

        else:
            return numpy.zeros(len(layout), dtype=numpy.bool_)

    out = apply(awkward1.operations.convert.tolayout(array, allowrecord=False))
    if highlevel:
        return awkward1._util.wrap(out,
                                   behavior=awkward1._util.behaviorof(array))
    else:
        return out

def notna(array, highlevel=True):
    """
    Args:
        array: Data to check for missing values (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Returns an array whose value is False where an element of `array` is None;
    True otherwise.

    See also #ak.isna, the logical negation of #ak.notna.
    """
    return ~isna(array, highlevel=highlevel)

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
    consider using #ak.tomask instead of a #ak.Array.__getitem__.

        >>> ak.tomask(array, ak.num(array) > 0)[:, 0]
        <Array [[1.1, 2.2, 3.3], None, [7.7]] type='3 * option[var * float64]'>
    """
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    out = layout.num(axis=axis)
    if highlevel:
        return awkward1._util.wrap(out,
                                   behavior=awkward1._util.behaviorof(array))
    else:
        return out

@awkward1._connect._numpy.implements(numpy.size)
def size(array, axis=None):
    """
    Args:
        array: Rectilinear array whose `shape` needs to be known.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
    Returns:
        An int or a list of ints, one for each regular dimension.

    Implements NumPy's
    [size](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html)
    function in a way that accepts #ak.Array as the `array`.

    If the `array` is not rectilinear (i.e. if #np.tonumpy would raise an
    error), then this function raise an error.
    """
    if axis is not None and axis < 0:
        raise NotImplementedError("ak.size with axis < 0")

    def recurse(layout, axis, sizes):
        if isinstance(layout, awkward1._util.unknowntypes):
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
                            "ak.size is ambiguous due to union of different "
                            "sizes")
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
                            "ak.size is ambiguous due to record of different "
                            "sizes")
            sizes.extend(compare)
        elif isinstance(layout, awkward1.layout.NumpyArray):
            if axis is None:
                sizes.extend(numpy.asarray(layout).shape[1:])
            else:
                sizes.extend(numpy.asarray(layout).shape[1:axis + 2])
        else:
            raise AssertionError("unrecognized Content type")

    layout = awkward1.operations.convert.tolayout(array, allowrecord=False)
    layout = awkward1.layout.RegularArray(layout, len(layout))

    sizes = []
    recurse(layout, axis, sizes)

    if axis is None:
        out = 1
        for size in sizes:
            if size is None:
                raise ValueError(
                        "ak.size is ambiguous due to variable-length arrays "
                        "(try ak.flatten to remove structure or ak.tonumpy to "
                        "force regularity, if possible)")
            else:
                out *= size
        return out
    else:
        if sizes[-1] is None:
            raise ValueError(
                    "ak.size is ambiguous due to variable-length arrays at "
                    "axis {0} (try ak.flatten to remove structure or "
                    "ak.tonumpy to force regularity, if possible)".format(
                                                                        axis))
        else:
            return sizes[-1]

@awkward1._connect._numpy.implements(numpy.atleast_1d)
def atleast_1d(*arrays):
    """
    Args:
        arrays: Rectilinear arrays to be converted to NumPy arrays of at
            least 1 dimension.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
    Returns:
        A NumPy array, not an Awkward Array.

    Implements NumPy's
    [atleast_1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html)
    function in a way that accepts #ak.Array objects as the `arrays`.

    If the `arrays` are not all rectilinear (i.e. if #np.tonumpy would raise an
    error), then this function raise an error.
    """
    return numpy.atleast_1d(*[awkward1.operations.convert.tonumpy(x)
                                for x in arrays])

@awkward1._connect._numpy.implements(numpy.concatenate)
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
    if axis != 0:
        raise NotImplementedError("axis={0}".format(axis))

    contents = [awkward1.operations.convert.tolayout(x, allowrecord=False)
                  for x in arrays]

    if len(contents) == 0:
        raise ValueError("need at least one array to concatenate")
    out = contents[0]
    for x in contents[1:]:
        if not out.mergeable(x, mergebool=mergebool):
            out = out.merge_as_union(x)
        else:
            out = out.merge(x)
        if isinstance(out, awkward1._util.uniontypes):
            out = out.simplify(mergebool=mergebool)

    if highlevel:
        return awkward1._util.wrap(out,
                                   behavior=awkward1._util.behaviorof(*arrays))
    else:
        return out

@awkward1._connect._numpy.implements(numpy.broadcast_arrays)
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
                                [1, 2, 3, 4, 5])
        [<Array [5, 5, 5, 5, 5] type='5 * int64'>,
         <Array [1, 2, 3, 4, 5] type='5 * int64'>]

    and

        >>> ak.broadcast_arrays(np.array([1, 2, 3]),
                                np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
        [<Array [[  1,   2,   3], [ 1,  2,  3]] type='2 * 3 * int64'>,
         <Array [[0.1, 0.2, 0.3], [10, 20, 30]] type='2 * 3 * float64'>]

    Note that in the second example, when the `3 * int64` array is expanded
    to match the `2 * 3 * float64` array, it is the deepest dimension that
    is aligned. If we try to match a `2 * int64` with the `2 * 3 * float64`,

        >>> ak.broadcast_arrays(np.array([1, 2]),
                                np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
        ValueError: cannot broadcast RegularArray of size 2 with RegularArray of size 3

    NumPy has the same behavior: arrays with different numbers of dimensions
    are aligned to the right before expansion. One can control this by
    explicitly adding a new axis (reshape to add a dimension of length 1)
    where the expansion is supposed to take place because a dimension of
    length 1 can be expanded like a scalar.

        >>> ak.broadcast_arrays(np.array([1, 2])[:, np.newaxis],
                                np.array([[0.1, 0.2, 0.3], [10, 20, 30]]))
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
                                [[1.1, 2.2, 3.3],    [], [4.4, 5.5]])
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
    highlevel, = awkward1._util.extra((), kwargs, [
        ("highlevel", True)])

    inputs = []
    for x in arrays:
        y = awkward1.operations.convert.tolayout(x,
                                                 allowrecord=True,
                                                 allowother=True)
        if not isinstance(y, (awkward1.layout.Content,
                              awkward1.layout.Record)):
            y = awkward1.layout.NumpyArray(numpy.array([y]))
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

@awkward1._connect._numpy.implements(numpy.where)
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
    mergebool, highlevel = awkward1._util.extra((), kwargs, [
        ("mergebool", True),
        ("highlevel", True)])

    npcondition = awkward1.operations.convert.tonumpy(condition)

    if len(args) == 0:
        out = numpy.nonzero(npcondition)
        if highlevel:
            return tuple(
                awkward1._util.wrap(awkward1.layout.NumpyArray(x),
                                    awkward1._util.behaviorof(condition))
                  for x in out)
        else:
            return tuple(awkward1.layout.NumpyArray(x) for x in out)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        if len(npcondition.shape) != 1:
            raise NotImplementedError(
                "FIXME: ak.where(condition, x, y) where condition is not 1-d")

        x = awkward1.operations.convert.tolayout(args[0], allowrecord=False)
        y = awkward1.operations.convert.tolayout(args[1], allowrecord=False)

        tags = (npcondition == 0)
        assert tags.itemsize == 1
        index = numpy.empty(len(tags), dtype=numpy.int64)
        index = numpy.arange(len(npcondition), dtype=numpy.int64)

        tags = awkward1.layout.Index8(tags.view(numpy.int8))
        index = awkward1.layout.Index64(index)
        tmp = awkward1.layout.UnionArray8_64(tags, index, [x, y])
        out = tmp.simplify(mergebool=mergebool)

        return awkward1._util.wrap(
                 out, behavior=awkward1._util.behaviorof(*((npcondition,)
                                                           + args)))

    else:
        raise TypeError(
                "where() takes from 1 to 3 positional arguments but {0} were "
                "given".format(len(args) + 1))

def flatten(array, axis=1, highlevel=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    out = layout.flatten(axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def rpad(array, length, axis=1, clip=False, highlevel=True):
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=False,
                                                  allowother=False)
    if clip:
        out = layout.rpad_and_clip(length, axis)
    else:
        out = layout.rpad(length, axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def fillna(array, value, highlevel=True):
    arraylayout = awkward1.operations.convert.tolayout(array,
                                                       allowrecord=True,
                                                       allowother=False)
    if isinstance(value, Iterable):
        valuelayout = awkward1.operations.convert.tolayout(value,
                                                           allowrecord=True,
                                                           allowother=False)
        if isinstance(valuelayout, awkward1.layout.Record):
            valuelayout = valuelayout.array[valuelayout.at:valuelayout.at + 1]
        elif len(valuelayout) == 0:
            offsets = awkward1.layout.Index64(numpy.array([0, 0],
                                                          dtype=numpy.int64))
            valuelayout = awkward1.layout.ListOffsetArray64(offsets,
                                                            valuelayout)
        else:
            valuelayout = awkward1.layout.RegularArray(valuelayout,
                                                       len(valuelayout))
    else:
        valuelayout = awkward1.operations.convert.tolayout([value],
                                                           allowrecord=True,
                                                           allowother=False)

    out = arraylayout.fillna(valuelayout)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def zip(arrays,
        depthlimit=None,
        parameters=None,
        withname=None,
        highlevel=True):
    if depthlimit is not None and depthlimit <= 0:
        raise ValueError("depthlimit must be None or at least 1")

    if isinstance(arrays, dict):
        recordlookup = []
        layouts = []
        for n, x in arrays.items():
            recordlookup.append(n)
            layouts.append(
                awkward1.operations.convert.tolayout(x,
                                                     allowrecord=False,
                                                     allowother=False))
    else:
        recordlookup = None
        layouts = []
        for x in arrays:
            layouts.append(
                awkward1.operations.convert.tolayout(x,
                                                     allowrecord=False,
                                                     allowother=False))

    if withname is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = withname

    def getfunction(inputs, depth):
        if ((depthlimit is None and
             any(x.purelist_depth == 1 for x in inputs)) or
            (depthlimit == depth)):
            return lambda: (
                awkward1.layout.RecordArray(inputs,
                                            recordlookup,
                                            parameters=parameters),)
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
    keys = awkward1.operations.describe.keys(array)
    if len(keys) == 0:
        return (array,)
    else:
        return tuple(array[n] for n in keys)

def argcross(arrays,
             axis=1,
             nested=None,
             parameters=None,
             withname=None,
             highlevel=True):
    if axis < 0:
        raise ValueError("argcross's 'axis' must be non-negative")

    else:
        if isinstance(arrays, dict):
            layouts = dict((n, awkward1.operations.convert.tolayout(
                                 x, allowrecord=False, allowother=False)
                               .localindex(axis))
                           for n, x in arrays.items())
        else:
            layouts = [awkward1.operations.convert.tolayout(
                         x,
                         allowrecord=False,
                         allowother=False).localindex(axis) for x in arrays]

        if withname is not None:
            if parameters is None:
                parameters = {}
            else:
                parameters = dict(parameters)
            parameters["__record__"] = withname

        result = cross(layouts,
                       axis=axis,
                       nested=nested,
                       parameters=parameters,
                       highlevel=False)

        if highlevel:
            return awkward1._util.wrap(result,
                                       awkward1._util.behaviorof(*arrays))
        else:
            return result

def cross(arrays,
          axis=1,
          nested=None,
          parameters=None,
          withname=None,
          highlevel=True):
    behavior = awkward1._util.behaviorof(*arrays)

    if withname is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = withname

    if axis < 0:
        raise ValueError("cross's 'axis' must be non-negative")

    elif axis == 0:
        if nested is None or nested is False:
            nested = []

        if isinstance(arrays, dict):
            if nested is True:
                nested = list(arrays.keys())   # last key is ignored below
            if any(not (isinstance(n, str) and n in arrays) for x in nested):
                raise ValueError(
                    "cross's 'nested' must be dict keys for a dict of arrays")
            recordlookup = []
            layouts = []
            tonested = []
            for i, (n, x) in enumerate(arrays.items()):
                recordlookup.append(n)
                layouts.append(
                    awkward1.operations.convert.tolayout(x,
                                                         allowrecord=False,
                                                         allowother=False))
                if n in nested:
                    tonested.append(i)
            nested = tonested

        else:
            if nested is True:
                nested = list(range(len(arrays) - 1))
            if any(not (isinstance(x, int) and 0 <= x < len(arrays) - 1)
                     for x in nested):
                raise ValueError(
                    "cross's 'nested' must be integers in [0, len(arrays) - 1)"
                    " for an iterable of arrays")
            recordlookup = None
            layouts = []
            for x in arrays:
                layouts.append(
                    awkward1.operations.convert.tolayout(x,
                                                         allowrecord=False,
                                                         allowother=False))

        indexes = [awkward1.layout.Index64(x.reshape(-1))
                     for x in numpy.meshgrid(*[numpy.arange(
                                               len(x),
                                               dtype=numpy.int64)
                                               for x in layouts],
                                             indexing="ij")]
        outs = [awkward1.layout.IndexedArray64(x, y)
                  for x, y in __builtins__["zip"](indexes, layouts)]

        result = awkward1.layout.RecordArray(outs,
                                             recordlookup,
                                             parameters=parameters)
        for i in range(len(arrays) - 1, -1, -1):
            if i in nested:
                result = awkward1.layout.RegularArray(result,
                                                      len(layouts[i + 1]))

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
                inside = len(arrays) - i - 1
                outside = i
                return lambda: newaxis(
                    awkward1._util.recursively_apply(layout,
                                                     getfunction1,
                                                     args=(inside,)), outside)
            else:
                return None

        def apply(x, i):
            return awkward1._util.recursively_apply(
                awkward1.operations.convert.tolayout(x,
                                                     allowrecord=False,
                                                     allowother=False),
                getfunction2,
                args=(i,))

        toflatten = []
        if nested is None or nested is False:
            nested = []

        if isinstance(arrays, dict):
            if nested is True:
                nested = list(arrays.keys())   # last key is ignored below
            if any(not (isinstance(n, str) and n in arrays) for x in nested):
                raise ValueError(
                    "cross's 'nested' must be dict keys for a dict of arrays")
            recordlookup = []
            layouts = []
            for i, (n, x) in enumerate(arrays.items()):
                recordlookup.append(n)
                layouts.append(apply(x, i))
                if i < len(arrays) - 1 and n not in nested:
                    toflatten.append(axis + i + 1)

        else:
            if nested is True:
                nested = list(range(len(arrays) - 1))
            if any(not (isinstance(x, int) and 0 <= x < len(arrays) - 1)
                   for x in nested):
                raise ValueError(
                    "cross's 'nested' must be integers in [0, len(arrays) - 1)"
                    " for an iterable of arrays")
            recordlookup = None
            layouts = []
            for i, x in enumerate(arrays):
                layouts.append(apply(x, i))
                if i < len(arrays) - 1 and i not in nested:
                    toflatten.append(axis + i + 1)

        def getfunction3(inputs, depth):
            if depth == axis + len(arrays):
                return lambda: (
                    awkward1.layout.RecordArray(inputs,
                                                recordlookup,
                                                parameters=parameters),)
            else:
                return None

        out = awkward1._util.broadcast_and_apply(layouts,
                                                 getfunction3,
                                                 behavior)
        assert isinstance(out, tuple) and len(out) == 1
        result = out[0]

        while len(toflatten) != 0:
            axis = toflatten.pop()
            result = flatten(result, axis=axis, highlevel=False)

    if highlevel:
        return awkward1._util.wrap(result, behavior)
    else:
        return result

def argchoose(array,
              n,
              diagonal=False,
              axis=1,
              keys=None,
              parameters=None,
              withname=None,
              highlevel=True):
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters)
    if withname is not None:
        parameters["__record__"] = withname

    if axis < 0:
        raise ValueError("argchoose's 'axis' must be non-negative")
    else:
        layout = awkward1.operations.convert.tolayout(
                   array, allowrecord=False, allowother=False).localindex(axis)
        out = layout.choose(n,
                            diagonal=diagonal,
                            keys=keys,
                            parameters=parameters,
                            axis=axis)
        if highlevel:
            return awkward1._util.wrap(
                out, behavior=awkward1._util.behaviorof(array))
        else:
            return out

def choose(array,
           n,
           diagonal=False,
           axis=1,
           keys=None,
           parameters=None,
           withname=None,
           highlevel=True):
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters)
    if withname is not None:
        parameters["__record__"] = withname

    layout = awkward1.operations.convert.tolayout(
               array, allowrecord=False, allowother=False)
    out = layout.choose(n,
                        diagonal=diagonal,
                        keys=keys,
                        parameters=parameters,
                        axis=axis)
    if highlevel:
        return awkward1._util.wrap(
                 out, behavior=awkward1._util.behaviorof(array))
    else:
        return out

def tomask(array, mask, validwhen=True, highlevel=True):
    def getfunction(inputs, depth):
        layoutarray, layoutmask = inputs
        if isinstance(layoutmask, awkward1.layout.NumpyArray):
            m = numpy.asarray(layoutmask)
            if not issubclass(m.dtype.type, (numpy.bool, numpy.bool_)):
                raise ValueError(
                    "mask must have boolean type, not "
                    "{0}".format(repr(m.dtype)))
            bytemask = awkward1.layout.Index8(m.view(numpy.int8))
            return lambda: (
                awkward1.layout.ByteMaskedArray(bytemask,
                                                layoutarray,
                                                validwhen=validwhen),)
        else:
            return None

    layoutarray = awkward1.operations.convert.tolayout(array,
                                                       allowrecord=True,
                                                       allowother=False)
    layoutmask = awkward1.operations.convert.tolayout(mask,
                                                      allowrecord=True,
                                                      allowother=False)

    behavior = awkward1._util.behaviorof(array, mask)
    out = awkward1._util.broadcast_and_apply([layoutarray, layoutmask],
                                             getfunction,
                                             behavior)
    assert isinstance(out, tuple) and len(out) == 1
    if highlevel:
        return awkward1._util.wrap(out[0], behavior)
    else:
        return out[0]

__all__ = [x for x in list(globals()) if not x.startswith("_") and
                                         x not in ("numpy", "awkward1")]
