# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._namedaxis import _get_named_axis, _unify_named_axis
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("unsafe_zip",)

np = NumpyMetadata.instance()


@high_level_function()
def unsafe_zip(
    arrays,
    *,
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

    Combines `arrays` into a single structure as the fields of a collection
    of records or the slots of a collection of tuples.

    Caution: unlike #ak.zip this function will _not_ broadcast the arrays together.
    It assumes that the given arrays have already the same layouts and lengths.

    This operation may be thought of as the opposite of projection in
    #ak.Array.__getitem__, which extracts fields one at a time, or
    #ak.unzip, which extracts them all in one call.

    Consider the following arrays, `one` and `two`.

        >>> one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
        >>> two = ak.Array([["a", "b", "c"], [], ["d", "e"], ["f"]])

    Zipping them together using a dict creates a collection of records with
    the same nesting structure as `one` and `two`.

        >>> ak.unsafe_zip({"x": one, "y": two}).show()
        [[{x: 1.1, y: 'a'}, {x: 2.2, y: 'b'}, {x: 3.3, y: 'c'}],
         [],
         [{x: 4.4, y: 'd'}],
         []]

    Doing so with a list creates tuples, whose fields are not named.

        >>> ak.zip([one, two]).show()
        [[(1.1, 'a'), (2.2, 'b'), (3.3, 'c')],
         [],
         [(4.4, 'd')],
         []]

    See also #ak.zip and #ak.unzip.
    """
    # Dispatch
    if isinstance(arrays, Mapping):
        yield arrays.values()
    else:
        yield arrays

    # Implementation
    return _impl(
        arrays,
        parameters,
        with_name,
        highlevel,
        behavior,
        attrs,
    )


def _impl(
    arrays,
    parameters,
    with_name,
    highlevel,
    behavior,
    attrs,
):
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

            # propagate named axis from input to output,
            #   use strategy "unify" (see: awkward._namedaxis)
            out_named_axis = reduce(
                _unify_named_axis, map(_get_named_axis, arrays.values())
            )

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

            # propagate named axis from input to output,
            #   use strategy "unify" (see: awkward._namedaxis)
            out_named_axis = reduce(_unify_named_axis, map(_get_named_axis, arrays))

    # determine backend
    backend = next((b.backend for b in layouts if hasattr(b, "backend")), "cpu")

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    # only allow all NumpyArrays and ListOffsetArrays
    # maybe this could be done recursively, but for now just check the top level. This is also how ak.zip works.
    if all(isinstance(layout, ak.contents.NumpyArray) for layout in layouts):
        length = layouts[0].length
        out = ak.contents.RecordArray(
            layouts, fields, length=length, parameters=parameters, backend=backend
        )
    elif all(isinstance(layout, ak.contents.ListOffsetArray) for layout in layouts):
        contents = []
        for layout in layouts:
            if not isinstance(layout.content, ak.contents.NumpyArray):
                raise ValueError(
                    "can not (unsafe) zip ListOffsetArrays with non-NumpyArray contents"
                )
            contents.append(layout.content)
        # just get from the first one
        offsets = layouts[0].offsets
        length = layouts[0].length
        out = ak.contents.ListOffsetArray(
            offsets=offsets,
            content=ak.contents.RecordArray(
                contents, fields, length=length, parameters=parameters, backend=backend
            ),
        )
    else:
        raise ValueError(
            "all array layouts must be either NumpyArrays or ListOffsetArrays"
        )

    # Unify named axes propagated through the broadcast
    wrapped_out = ctx.wrap(out, highlevel=highlevel)
    return ak.operations.ak_with_named_axis._impl(
        wrapped_out,
        named_axis=out_named_axis,
        highlevel=highlevel,
        behavior=ctx.behavior,
        attrs=ctx.attrs,
    )
