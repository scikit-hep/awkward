# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("where",)
import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@ak._connect.numpy.implements("where")
@high_level_function()
def where(condition, *args, mergebool=True, highlevel=True, behavior=None):
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
        return _impl1(condition, mergebool, highlevel, behavior)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        x, y = args
        return _impl3(condition, x, y, mergebool, highlevel, behavior)

    else:
        raise TypeError(
            f"where() takes from 1 to 3 positional arguments but {len(args) + 1} were "
            "given"
        )


def _impl1(condition, mergebool, highlevel, behavior):
    akcondition = ak.operations.to_layout(
        condition, allow_record=False, allow_other=False
    )
    backend = backend_of(akcondition, default=cpu)

    akcondition = ak.contents.NumpyArray(ak.operations.to_numpy(akcondition))
    out = backend.nplike.nonzero(ak.operations.to_numpy(akcondition))
    if highlevel:
        return tuple(
            wrap_layout(
                ak.contents.NumpyArray(x),
                behavior_of(condition, behavior=behavior),
            )
            for x in out
        )
    else:
        return tuple(ak.contents.NumpyArray(x) for x in out)


def _impl3(condition, x, y, mergebool, highlevel, behavior):
    behavior = behavior_of(condition, x, y, behavior=behavior)
    backend = backend_of(condition, x, y, default=cpu, coerce_to_common=True)

    condition_layout = ak.operations.to_layout(
        condition, allow_record=False, allow_other=False
    ).to_backend(backend)

    x_layout = ak.operations.to_layout(x, allow_record=False, allow_other=True)
    if isinstance(x_layout, ak.contents.Content):
        x_layout = x_layout.to_backend(backend)

    y_layout = ak.operations.to_layout(y, allow_record=False, allow_other=True)
    if isinstance(y_layout, ak.contents.Content):
        y_layout = y_layout.to_backend(backend)

    def action(inputs, **kwargs):
        condition, x, y = inputs
        if isinstance(condition, ak.contents.NumpyArray):
            npcondition = backend.index_nplike.asarray(condition)
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

    out = ak._broadcasting.broadcast_and_apply(
        [condition_layout, x_layout, y_layout],
        action,
        behavior,
        numpy_to_regular=True,
    )

    return wrap_layout(out[0], behavior, highlevel)
