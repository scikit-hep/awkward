# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("pad_none",)

np = NumpyMetadata.instance()


@high_level_function()
def pad_none(
    array, target, axis=1, *, clip=False, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
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
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Increase the lengths of lists to a target length by adding None values.

    Consider the following

        >>> array = ak.Array([[[1.1, 2.2, 3.3],
        ...                    [],
        ...                    [4.4, 5.5],
        ...                    [6.6]],
        ...                   [],
        ...                   [[7.7],
        ...                    [8.8, 9.9]
        ...                   ]])

    At `axis=0`, this operation pads the whole array, adding None at the
    outermost level:

        >>> ak.pad_none(array, 5, axis=0).show()
        [[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]],
         [],
         [[7.7], [8.8, 9.9]],
         None,
         None]

    At `axis=1`, this operation pads the first nested level:

        >>> ak.pad_none(array, 3, axis=1).show()
        [[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]],
         [None, None, None],
         [[7.7], [8.8, 9.9], None]]

    And so on for higher values of `axis`:

        >>> ak.pad_none(array, 2, axis=2).show()
        [[[1.1, 2.2, 3.3], [None, None], [4.4, 5.5], [6.6, None]],
         [],
         [[7.7, None], [8.8, 9.9]]]

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
        <Array [[[1.1, 2.2, 3.3], ..., [...]], ...] type='3 * var * var * ?float64'>

    and

        >>> ak.pad_none(array, 2, axis=2, clip=True)
        <Array [[[1.1, 2.2], ..., [6.6, None]], ...] type='3 * var * 2 * ?float64'>

    is not just in the length of `[1.1, 2.2, 3.3]` vs `[1.1, 2.2]`, but also
    in the distinction between the following types.

        >>> ak.pad_none(array, 2, axis=2).type.show()
        3 * var * var * ?float64
        >>> ak.pad_none(array, 2, axis=2, clip=True).type.show()
        3 * var *   2 * ?float64
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, target, axis, clip, highlevel, behavior, attrs)


def _impl(array, target, axis, clip, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    out = ak._do.pad_none(layout, target, axis, clip=clip)

    return ctx.wrap(out, highlevel=highlevel)
