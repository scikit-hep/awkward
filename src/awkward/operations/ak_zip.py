# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("zip",)

np = NumpyMetadata.instance()


@high_level_function()
def zip(
    arrays,
    depth_limit=None,
    *,
    parameters=None,
    with_name=None,
    right_broadcast=False,
    optiontype_outside_record=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        arrays (mapping or sequence of arrays): Each value in this mapping or
            sequence can be any array-like data that #ak.to_layout recognizes.
        depth_limit (None or int): If None, attempt to fully broadcast the
            `array` to all levels. If an int, limit the number of dimensions
            that get broadcasted. The minimum value is `1`, for no
            broadcasting.
        parameters (None or dict): Parameters for the new
            #ak.contents.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.contents.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        right_broadcast (bool): If True, follow rules for implicit
            right-broadcasting, as described in #ak.broadcast_arrays.
        optiontype_outside_record (bool): If True, continue broadcasting past
            any option types before creating the new #ak.contents.RecordArray node.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

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

        >>> ak.zip({"x": one, "y": two}).show()
        [[{x: 1.1, y: 'a'}, {x: 2.2, y: 'b'}, {x: 3.3, y: 'c'}],
         [],
         [{x: 4.4, y: 'd'}, {x: 5.5, y: 'e'}],
         [{x: 6.6, y: 'f'}]]

    Doing so with a list creates tuples, whose fields are not named.

        >>> ak.zip([one, two]).show()
        [[(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
         [],
         [(4.4, 'd'), (5.5, 'e')],
         [(6.6, 'f')]]

    Adding a third array with the same length as `one` and `two` but less
    internal structure is okay: it gets broadcasted to match the others.
    (See #ak.broadcast_arrays for broadcasting rules.)

        >>> three = ak.Array([100, 200, 300, 400])
        >>> ak.zip([one, two, three]).show()
        [[(1.1, 'a', 100), (2.2, 'b', 100), (3.3, 'c', 100)],
         [],
         [(4.4, 'd', 300), (5.5, 'e', 300)],
         [(6.6, 'f', 400)]]

    However, if arrays have the same depth but different lengths of nested
    lists, attempting to zip them together is a broadcasting error.

        >>> one = ak.Array([[[1, 2, 3], [], [4, 5], [6]], [], [[7, 8]]])
        >>> two = ak.Array([[[1.1, 2.2], [3.3], [4.4], [5.5]], [], [[6.6]]])
        >>> ak.zip([one, two])
        ValueError: while calling
            ak.zip(
                arrays = [<Array [[[1, 2, 3], [], [4, ...], [6]], ...] type='3 * var ...
                depth_limit = None
                parameters = None
                with_name = None
                right_broadcast = False
                optiontype_outside_record = False
                highlevel = True
                behavior = None
            )
        Error details: cannot broadcast nested list

    For this, one can set the `depth_limit` to prevent the operation from
    attempting to broadcast what can't be broadcasted.

        >>> ak.zip([one, two], depth_limit=1).show()
        [([[1, 2, 3], [], [4, ...], [6]], [[1.1, ...], ...]),
         ([], []),
         ([[7, 8]], [[6.6]])]

    As an extreme, `depth_limit=1` is a handy way to make a record structure
    at the outermost level, regardless of whether the fields have matching
    structure or not.

    When zipping together arrays with optional values, it can be useful to create
    the #ak.contents.RecordArray node after the option types. By default, #ak.zip
    does not do this:

        >>> one = ak.Array([1, 2, None])
        >>> two = ak.Array([None, 5, 6])
        >>> ak.zip([one, two])
        <Array [(1, None), (2, 5), (None, 6)] type='3 * (?int64, ?int64)'>

    If the `optiontype_outside_record` option is set to `True`, Awkward will continue to
    broadcast the arrays together at the depth_limit until it reaches non-option
    types. This effectively takes the union of the option mask:

        >>> ak.zip([one, two], optiontype_outside_record=True)
        <Array [None, (2, 5), None] type='3 * ?(int64, int64)'>
    """
    # Dispatch
    if isinstance(arrays, Mapping):
        yield arrays.values()
    else:
        yield arrays

    # Implementation
    return _impl(
        arrays,
        depth_limit,
        parameters,
        with_name,
        right_broadcast,
        optiontype_outside_record,
        highlevel,
        behavior,
        attrs,
    )


def _impl(
    arrays,
    depth_limit,
    parameters,
    with_name,
    right_broadcast,
    optiontype_outside_record,
    highlevel,
    behavior,
    attrs,
):
    if depth_limit is not None and depth_limit <= 0:
        raise ValueError("depth_limit must be None or at least 1")
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        if isinstance(arrays, Mapping):
            layouts = ensure_same_backend(
                *(
                    ctx.unwrap(
                        x,
                        allow_record=False,
                        allow_unknown=False,
                        none_policy="pass-through",
                        primitive_policy="pass-through",
                    )
                    for x in arrays.values()
                )
            )
            fields = list(arrays.keys())

        else:
            layouts = ensure_same_backend(
                *(
                    ctx.unwrap(
                        x,
                        allow_record=False,
                        allow_unknown=False,
                        none_policy="pass-through",
                        primitive_policy="pass-through",
                    )
                    for x in arrays
                )
            )
            fields = None

    # Promote any integers or records
    backend = next((b.backend for b in layouts if hasattr(b, "backend")), "cpu")
    layout_is_content = [isinstance(x, ak.contents.Content) for x in layouts]
    layouts = [
        x
        if isinstance(x, (ak.contents.Content, ak.record.Record))
        else ak.operations.to_layout(x).to_backend(backend)
        for x in layouts
    ]

    to_record = not any(layout_is_content)

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    def action(inputs, depth, backend, **ignore):
        if depth_limit == depth or all(x.purelist_depth == 1 for x in inputs):
            # If we want to zip after option types at this depth
            if optiontype_outside_record and any(x.is_option for x in inputs):
                return None

            return (
                ak.contents.RecordArray(
                    inputs, fields, parameters=parameters, backend=backend
                ),
            )
        else:
            return None

    out = ak._broadcasting.broadcast_and_apply(
        layouts, action, right_broadcast=right_broadcast
    )
    assert isinstance(out, tuple) and len(out) == 1
    out = out[0]

    if to_record:
        out = out[0]
        assert isinstance(out, ak.record.Record)

    return ctx.wrap(out, highlevel=highlevel)
