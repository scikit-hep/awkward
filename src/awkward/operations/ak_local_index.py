# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import regularize_axis

__all__ = ("local_index",)

np = NumpyMetadata.instance()


@high_level_function()
def local_index(array, axis=-1, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

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
        <Array [[[0, 1, 2], []], ..., [[0], ..., [...]]] type='4 * var * var * int64'>

    Note that you can make a Pandas-style MultiIndex by calling this function on
    every axis.

        >>> multiindex = ak.zip([ak.local_index(array, i) for i in range(array.ndim)])
        >>> multiindex.show()
        [[[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
         [[(1, 0, 0), (1, 0, 1)]],
         [],
         [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]]]
        >>> ak.flatten(ak.flatten(multiindex)).show()
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

    But if you're interested in Pandas, you may want to use #ak.to_dataframe directly.

        >>> ak.to_dataframe(array)
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
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    out = ak._do.local_index(layout, axis)
    return ctx.wrap(out, highlevel=highlevel)
