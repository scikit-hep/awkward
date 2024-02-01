# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("cartesian",)

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@high_level_function()
def cartesian(
    arrays,
    axis=1,
    *,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        arrays (mapping or sequence of arrays): Each value in this mapping or
            sequence can be any array-like data that #ak.to_layout recognizes.
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
        attrs (None or dict): Custom attributes for the output array, if
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
    order that the `arrays` are written.

    To emulate an SQL or Pandas "group by" operation, put the keys that you
    wish to group by *first* and use `nested=[0]` or `nested=[n]` to group by
    unique n-tuples. If necessary, record keys can later be reordered with a
    list of strings in #ak.Array.__getitem__.

    To get list index positions in the tuples/records, rather than data from
    the original `arrays`, use #ak.argcartesian instead of #ak.cartesian. The
    #ak.argcartesian form can be particularly useful as nested indexing in
    #ak.Array.__getitem__.
    """
    # Dispatch
    if isinstance(arrays, Mapping):
        yield arrays.values()
    else:
        yield arrays

    # Implementation
    return _impl(
        arrays, axis, nested, parameters, with_name, highlevel, behavior, attrs
    )


def _impl(arrays, axis, nested, parameters, with_name, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        if isinstance(arrays, Mapping):
            layouts = ensure_same_backend(
                *(
                    ctx.unwrap(x, allow_record=False, allow_unknown=False)
                    for x in arrays.values()
                )
            )
            fields = list(arrays.keys())
            array_layouts = dict(zip(fields, layouts))

        else:
            layouts = array_layouts = ensure_same_backend(
                *(
                    ctx.unwrap(x, allow_record=False, allow_unknown=False)
                    for x in arrays
                )
            )
            fields = None

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    posaxis = maybe_posaxis(layouts[0], axis, 1)
    # Validate `posaxis`
    if posaxis is None or posaxis < 0:
        raise AxisError("negative axis depth is ambiguous")
    # Ensure other layouts have same positive value for axis
    for layout in layouts[1:]:
        if maybe_posaxis(layout, axis, 1) != posaxis:
            raise AxisError(
                "arrays to cartesian-product do not have the same depth for negative axis"
            )
    depths = [obj.purelist_depth for obj in layouts]
    if posaxis >= max(depths):
        raise AxisError(
            f"axis={axis} exceeds the max depth of the given arrays (which is {max(depths)})"
        )

    # Validate `nested`
    if nested is None or nested is False:
        nested = []
    elif nested is True:
        if fields is not None:
            nested = list(fields)[:-1]
        else:
            nested = list(range(len(layouts))[:-1])
    else:
        if isinstance(array_layouts, Mapping):
            if any(not (isinstance(x, str) and x in array_layouts) for x in nested):
                raise ValueError(
                    "the 'nested' parameter of cartesian must be dict keys "
                    "for a dict of arrays"
                )
            if len(nested) >= len(array_layouts):
                raise ValueError(
                    "the `nested` parameter of cartesian must contain "
                    "fewer items than there are arrays"
                )
        else:
            if any(
                not (isinstance(x, int) and 0 <= x < len(array_layouts) - 1)
                for x in nested
            ):
                raise ValueError(
                    "the 'nested' parameter of cartesian must be integers in "
                    "[0, len(arrays) - 1) for an iterable of arrays"
                )

    backend = next((layout.backend for layout in layouts), cpu)
    if posaxis == 0:
        # Translate nested field names to nested field indices
        if fields is not None:
            nested_as_index = [i for i, name in enumerate(fields) if name in nested]
        else:
            nested_as_index = nested

        indexes = [
            ak.index.Index64(backend.index_nplike.reshape(x, (-1,)))
            for x in backend.index_nplike.meshgrid(
                *[
                    backend.index_nplike.arange(x.length, dtype=np.int64)
                    for x in layouts
                ],
                indexing="ij",
            )
        ]
        outs = [
            ak.contents.IndexedArray.simplified(x, y) for x, y in zip(indexes, layouts)
        ]

        result = ak.contents.RecordArray(outs, fields, parameters=parameters)
        for i in range(len(array_layouts))[::-1]:
            if i in nested_as_index:
                result = ak.contents.RegularArray(result, layouts[i + 1].length, 0)

    else:
        # Translate nested field names to nested field indices
        if fields is not None:
            nested_as_index = [i for i, name in enumerate(fields) if name in nested]
        else:
            nested_as_index = nested

        def add_outer_dimensions(
            layout: ak.contents.Content, n: int
        ) -> ak.contents.Content:
            if n == 0:
                return layout
            else:
                return ak.contents.RegularArray(
                    add_outer_dimensions(layout, n - 1), 1, 0
                )

        def apply_pad_inner_list(layout, depth, lateral_context, **kwargs):
            """
            Add new dimensions (given by lateral_context["n"]) above innermost list
            """
            n = lateral_context["n"]
            # We want to be above at least one dimension (list)
            if depth == 2:
                return add_outer_dimensions(layout, n)
            else:
                return None

        def apply_pad_inner_list_at_axis(layout, depth, lateral_context, **kwargs):
            """
            Each array in arrays contributes to one of these new dimensions.
            To make the cartesian product of the given arrays broadcastable,
            each array is padded by (n, m) new length-1 regular dimensions
            (above, below) the target depth. The values of (n, m) are given by
            the position of the array; the first array is the outermost axis.
            """
            i = lateral_context["i"]
            if depth == posaxis:
                n_inside = len(array_layouts) - i - 1
                n_outside = i
                if (
                    layout.parameter("__array__") == "string"
                    or layout.parameter("__array__") == "bytestring"
                ):
                    raise ValueError(
                        "ak.cartesian does not compute combinations of the "
                        "characters of a string; please split it into lists"
                    )
                nextlayout = ak._do.recursively_apply(
                    layout,
                    apply_pad_inner_list,
                    lateral_context={"n": n_inside},
                )
                return add_outer_dimensions(nextlayout, n_outside)
            else:
                return None

        # New _interior_ axes are added to the result layout, but
        # unless explicitly named, these axes should be flattened.
        axes_to_flatten = [
            posaxis + i + 1
            for i, _ in enumerate(array_layouts)
            if i < len(array_layouts) - 1 and i not in nested_as_index
        ]
        # This list *must* be sorted in reverse order
        axes_to_flatten.reverse()

        new_layouts = [
            ak._do.recursively_apply(
                layout,
                apply_pad_inner_list_at_axis,
                lateral_context={"i": i},
            )
            for i, layout in enumerate(layouts)
        ]

        def apply_build_record(inputs, depth, **kwargs):
            if depth == posaxis + len(array_layouts):
                return (ak.contents.RecordArray(inputs, fields, parameters=parameters),)

            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            new_layouts, apply_build_record, right_broadcast=False
        )
        assert isinstance(out, tuple) and len(out) == 1
        result = out[0]

        # Remove surplus dimensions, iterating from smallest to greatest
        for axis_to_flatten in axes_to_flatten:
            result = ak.operations.flatten(
                result, axis=axis_to_flatten, highlevel=False, behavior=behavior
            )

    return ctx.wrap(result, highlevel=highlevel)
