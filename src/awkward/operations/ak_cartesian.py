# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def cartesian(
    arrays,
    axis=1,
    *,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        arrays (dict or iterable of arrays): Each value in this dict or iterable
            can be any array-like data that #ak.to_layout recognizes.
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
            #ak.contents.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.contents.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Computes a Cartesian product (i.e. cross product) of data from a set of
    `arrays`. This operation creates records (if `arrays` is a dict) or tuples
    (if `arrays` is another kind of iterable) that hold the combinations
    of elements, and it can introduce new levels of nesting.

    As a simple example with `axis=0`, the Cartesian product of

        >>> one = ak.Array([1, 2, 3])
        >>> two = ak.Array(["a", "b"])

    is

        >>> ak.cartesian([one, two], axis=0).show()
        [(1, 'a'),
         (1, 'b'),
         (2, 'a'),
         (2, 'b'),
         (3, 'a'),
         (3, 'b')]

    With nesting, a new level of nested lists is created to group combinations
    that share the same element from `one` into the same list.

        >>> ak.cartesian([one, two], axis=0, nested=True).show()
        [[(1, 'a'), (1, 'b')],
         [(2, 'a'), (2, 'b')],
         [(3, 'a'), (3, 'b')]]

    The primary purpose of this function, however, is to compute a different
    Cartesian product for each element of an array: in other words, `axis=1`.
    The following arrays each have four elements.

        >>> one = ak.Array([[1, 2, 3], [], [4, 5], [6]])
        >>> two = ak.Array([["a", "b"], ["c"], ["d"], ["e", "f"]])

    The default `axis=1` produces 6 pairs from the Cartesian product of
    `[1, 2, 3]` and `["a", "b"]`, 0 pairs from `[]` and `["c"]`, 1 pair from
    `[4, 5]` and `["d"]`, and 1 pair from `[6]` and `["e", "f"]`.

        >>> ak.cartesian([one, two]).show()
        [[(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')],
         [],
         [(4, 'd'), (5, 'd')],
         [(6, 'e'), (6, 'f')]]

    The nesting depth is the same as the original arrays; with `nested=True`,
    the nesting depth is increased by 1 and tuples are grouped by their
    first element.

        >>> ak.cartesian([one, two], nested=True).show()
        [[[(1, 'a'), (1, 'b')], [(2, 'a'), (2, ...)], [(3, 'a'), (3, 'b')]],
         [],
         [[(4, 'd')], [(5, 'd')]],
         [[(6, 'e'), (6, 'f')]]]

    These tuples are #ak.contents.RecordArray nodes with unnamed fields. To
    name the fields, we can pass `one` and `two` in a dict, rather than a list.

        >>> ak.cartesian({"x": one, "y": two}).show()
        [[{x: 1, y: 'a'}, {x: 1, y: 'b'}, {...}, ..., {x: 3, y: 'a'}, {x: 3, y: 'b'}],
         [],
         [{x: 4, y: 'd'}, {x: 5, y: 'd'}],
         [{x: 6, y: 'e'}, {x: 6, y: 'f'}]]

    With more than two elements in the Cartesian product, `nested` can specify
    which are grouped and which are not. For example,

        >>> one = ak.Array([1, 2, 3, 4])
        >>> two = ak.Array([1.1, 2.2, 3.3])
        >>> three = ak.Array(["a", "b"])

    can be left entirely ungrouped:

        >>> ak.cartesian([one, two, three], axis=0).show()
        [(1, 1.1, 'a'),
         (1, 1.1, 'b'),
         (1, 2.2, 'a'),
         (1, 2.2, 'b'),
         (1, 3.3, 'a'),
         (1, 3.3, 'b'),
         (2, 1.1, 'a'),
         (2, 1.1, 'b'),
         (2, 2.2, 'a'),
         (2, 2.2, 'b'),
         ...,
         (3, 2.2, 'b'),
         (3, 3.3, 'a'),
         (3, 3.3, 'b'),
         (4, 1.1, 'a'),
         (4, 1.1, 'b'),
         (4, 2.2, 'a'),
         (4, 2.2, 'b'),
         (4, 3.3, 'a'),
         (4, 3.3, 'b')]

    can be grouped by `one` (adding 1 more dimension):

        >>> ak.cartesian([one, two, three], axis=0, nested=[0]).show()
        [[(1, 1.1, 'a'), (1, 1.1, 'b'), (1, 2.2, 'a')],
         [(1, 2.2, 'b'), (1, 3.3, 'a'), (1, 3.3, 'b')],
         [(2, 1.1, 'a'), (2, 1.1, 'b'), (2, 2.2, 'a')],
         [(2, 2.2, 'b'), (2, 3.3, 'a'), (2, 3.3, 'b')],
         [(3, 1.1, 'a'), (3, 1.1, 'b'), (3, 2.2, 'a')],
         [(3, 2.2, 'b'), (3, 3.3, 'a'), (3, 3.3, 'b')],
         [(4, 1.1, 'a'), (4, 1.1, 'b'), (4, 2.2, 'a')],
         [(4, 2.2, 'b'), (4, 3.3, 'a'), (4, 3.3, 'b')]]

    can be grouped by `one` and `two` (adding 2 more dimensions):

        >>> ak.cartesian([one, two, three], axis=0, nested=[0, 1]).show()
        [[[(1, 1.1, 'a'), (1, 1.1, 'b')], [...], [(1, 3.3, 'a'), (1, 3.3, ...)]],
         [[(2, 1.1, 'a'), (2, 1.1, 'b')], [...], [(2, 3.3, 'a'), (2, 3.3, ...)]],
         [[(3, 1.1, 'a'), (3, 1.1, 'b')], [...], [(3, 3.3, 'a'), (3, 3.3, ...)]],
         [[(4, 1.1, 'a'), (4, 1.1, 'b')], [...], [(4, 3.3, 'a'), (4, 3.3, ...)]]]

    or grouped by unique `one`-`two` pairs (adding 1 more dimension):

        >>> ak.cartesian([one, two, three], axis=0, nested=[1]).show()
        [[(1, 1.1, 'a'), (1, 1.1, 'b')],
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
         [(4, 3.3, 'a'), (4, 3.3, 'b')]]

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
    with ak._errors.OperationErrorContext(
        "ak.cartesian",
        dict(
            arrays=arrays,
            axis=axis,
            nested=nested,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(arrays, axis, nested, parameters, with_name, highlevel, behavior)


def _impl(arrays, axis, nested, parameters, with_name, highlevel, behavior):
    if isinstance(arrays, dict):
        behavior = ak._util.behavior_of(*arrays.values(), behavior=behavior)
        backend = ak._backends.backend_of(*arrays.values(), default=cpu)
        new_arrays = {}
        for n, x in arrays.items():
            new_arrays[n] = ak.operations.to_layout(
                x, allow_record=False, allow_other=False
            )

    else:
        arrays = list(arrays)
        behavior = ak._util.behavior_of(*arrays, behavior=behavior)
        backend = ak._backends.backend_of(*arrays, default=cpu)
        new_arrays = []
        for x in arrays:
            new_arrays.append(
                ak.operations.to_layout(x, allow_record=False, allow_other=False)
            )

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    if isinstance(new_arrays, dict):
        new_arrays_values = list(new_arrays.values())
    else:
        new_arrays_values = new_arrays

    posaxis = ak._util.maybe_posaxis(new_arrays_values[0], axis, 1)
    if posaxis is None or posaxis < 0:
        raise ak._errors.wrap_error(ValueError("negative axis depth is ambiguous"))
    for x in new_arrays_values[1:]:
        if ak._util.maybe_posaxis(x, axis, 1) != posaxis:
            raise ak._errors.wrap_error(
                ValueError(
                    "arrays to cartesian-product do not have the same depth for negative axis"
                )
            )

    if posaxis == 0:
        if nested is None or nested is False:
            nested = []

        if isinstance(new_arrays, dict):
            if nested is True:
                nested = list(new_arrays.keys())  # last key is ignored below
            if any(not (isinstance(n, str) and n in new_arrays) for x in nested):
                raise ak._errors.wrap_error(
                    ValueError(
                        "the 'nested' parameter of cartesian must be dict keys "
                        "for a dict of arrays"
                    )
                )
            fields = []
            layouts = []
            tonested = []
            for i, (n, x) in enumerate(new_arrays.items()):
                fields.append(n)
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
                raise ak._errors.wrap_error(
                    ValueError(
                        "the 'nested' prarmeter of cartesian must be integers in "
                        "[0, len(arrays) - 1) for an iterable of arrays"
                    )
                )
            fields = None
            layouts = []
            for x in new_arrays:
                layouts.append(x)

        layouts = [x for x in layouts]

        indexes = [
            ak.index.Index64(x.reshape(-1))
            for x in backend.index_nplike.meshgrid(
                *[backend.index_nplike.arange(len(x), dtype=np.int64) for x in layouts],
                indexing="ij",
            )
        ]
        outs = [
            ak.contents.IndexedArray.simplified(x, y)
            for x, y in __builtins__["zip"](indexes, layouts)
        ]

        result = ak.contents.RecordArray(outs, fields, parameters=parameters)
        for i in range(len(new_arrays) - 1, -1, -1):
            if i in nested:
                result = ak.contents.RegularArray(result, len(layouts[i + 1]), 0)

    else:

        def newaxis(layout, i):
            if i == 0:
                return layout
            else:
                return ak.contents.RegularArray(newaxis(layout, i - 1), 1, 0)

        def getgetfunction1(i, **kwargs):
            def getfunction1(layout, depth, **kwargs):
                if depth == 2:
                    return newaxis(layout, i)
                else:
                    return None

            return getfunction1

        def getgetfunction2(i, **kwargs):
            def getfunction2(layout, depth, **kwargs):
                if depth == posaxis:
                    inside = len(new_arrays) - i - 1
                    outside = i
                    if (
                        layout.parameter("__array__") == "string"
                        or layout.parameter("__array__") == "bytestring"
                    ):
                        raise ak._errors.wrap_error(
                            ValueError(
                                "ak.cartesian does not compute combinations of the "
                                "characters of a string; please split it into lists"
                            )
                        )
                    nextlayout = ak._do.recursively_apply(
                        layout, getgetfunction1(inside), behavior
                    )
                    return newaxis(nextlayout, outside)
                else:
                    return None

            return getfunction2

        def apply(x, i):
            layout = ak.operations.to_layout(x, allow_record=False, allow_other=False)
            return ak._do.recursively_apply(layout, getgetfunction2(i), behavior)

        toflatten = []
        if nested is None or nested is False:
            nested = []

        if isinstance(new_arrays, dict):
            if nested is True:
                nested = list(new_arrays.keys())  # last key is ignored below
            if any(not (isinstance(n, str) and n in new_arrays) for x in nested):
                raise ak._errors.wrap_error(
                    ValueError(
                        "the 'nested' parameter of cartesian must be dict keys "
                        "for a dict of arrays"
                    )
                )
            fields = []
            layouts = []
            for i, (n, x) in enumerate(new_arrays.items()):
                fields.append(n)
                layouts.append(apply(x, i))
                if i < len(new_arrays) - 1 and n not in nested:
                    toflatten.append(posaxis + i + 1)

        else:
            if nested is True:
                nested = list(range(len(new_arrays) - 1))
            if any(
                not (isinstance(x, int) and 0 <= x < len(new_arrays) - 1)
                for x in nested
            ):
                raise ak._errors.wrap_error(
                    ValueError(
                        "the 'nested' parameter of cartesian must be integers in "
                        "[0, len(arrays) - 1) for an iterable of arrays"
                    )
                )
            fields = None
            layouts = []
            for i, x in enumerate(new_arrays):
                layouts.append(apply(x, i))
                if i < len(new_arrays) - 1 and i not in nested:
                    toflatten.append(posaxis + i + 1)

        def getfunction3(inputs, depth, **kwargs):
            if depth == posaxis + len(new_arrays):
                if all(len(x) == 0 for x in inputs):
                    inputs = [
                        x.content
                        if isinstance(x, ak.contents.RegularArray) and x.size == 1
                        else x
                        for x in inputs
                    ]
                return (ak.contents.RecordArray(inputs, fields, parameters=parameters),)

            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            layouts, getfunction3, behavior, right_broadcast=False
        )
        assert isinstance(out, tuple) and len(out) == 1
        result = out[0]

        while len(toflatten) != 0:
            flatten_axis = toflatten.pop()
            result = ak.operations.flatten(
                result, axis=flatten_axis, highlevel=False, behavior=behavior
            )

    return ak._util.wrap(result, behavior, highlevel)
