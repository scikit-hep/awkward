# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("where",)

np = NumpyMetadata.instance()


@ak._connect.numpy.implements("where")
@high_level_function()
def where(condition, *args, mergebool=True, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        condition: Array-like data (anything #ak.to_layout recognizes) of booleans.
        x: Optional array-like data (anything #ak.to_layout recognizes) with the same
            length as `condition`.
        y: Optional array-like data (anything #ak.to_layout recognizes) with the same
            length as `condition`.
        mergebool (bool, default is True): If True, boolean and numeric data
            can be combined into the same buffer, losing information about
            False vs `0` and True vs `1`; otherwise, they are kept in separate
            buffers with distinct types (using an #ak.contents.UnionArray).
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

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
    # Dispatch
    yield (*args, condition)

    # Implementation
    if len(args) == 0:
        return _impl1(condition, mergebool, highlevel, behavior, attrs)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        x, y = args
        return _impl3(condition, x, y, mergebool, highlevel, behavior, attrs)

    else:
        raise TypeError(
            f"where() takes from 1 to 3 positional arguments but {len(args) + 1} were "
            "given"
        )


def _impl1(condition, mergebool, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(condition, allow_record=False, primitive_policy="error")
    out = layout.backend.nplike.nonzero(layout.to_backend_array(allow_missing=False))

    return tuple(
        ctx.wrap(ak.contents.NumpyArray(x, backend=layout.backend), highlevel=highlevel)
        for x in out
    )


def _impl3(condition, x, y, mergebool, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layouts = ensure_same_backend(
            ctx.unwrap(x, allow_record=False, primitive_policy="pass-through"),
            ctx.unwrap(y, allow_record=False, primitive_policy="pass-through"),
            ctx.unwrap(condition, allow_record=False, primitive_policy="error"),
        )

    def action(inputs, backend, **kwargs):
        x, y, condition = inputs
        if isinstance(condition, ak.contents.NumpyArray):
            npcondition = backend.index_nplike.asarray(condition.data)
            tags = ak.index.Index8((npcondition == 0).view(np.int8))
            index = ak.index.Index64(
                backend.index_nplike.arange(tags.length, dtype=np.int64),
                nplike=backend.index_nplike,
            )
            if not isinstance(x, ak.contents.Content):
                x = ak.contents.NumpyArray(
                    backend.nplike.repeat(
                        backend.nplike.asarray(x),
                        backend.nplike.shape_item_as_index(tags.length),
                    )
                )
            if not isinstance(y, ak.contents.Content):
                y = ak.contents.NumpyArray(
                    backend.nplike.repeat(
                        backend.nplike.asarray(y),
                        backend.nplike.shape_item_as_index(tags.length),
                    )
                )
            return (
                ak.contents.UnionArray.simplified(
                    tags,
                    index,
                    [x, y],
                    mergebool=mergebool,
                ),
            )
        else:
            return None

    out = ak._broadcasting.broadcast_and_apply(layouts, action, numpy_to_regular=True)

    return ctx.wrap(out[0], highlevel=highlevel)
