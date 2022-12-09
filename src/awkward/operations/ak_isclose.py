# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("isclose")
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, *, highlevel=True, behavior=None
):
    """
    Args:
        a: Array-like data (anything #ak.to_layout recognizes).
        b: Array-like data (anything #ak.to_layout recognizes).
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        equal_nan (bool): Whether to compare `NaN` as equal. If True, `NaN` in
            `a` will be considered equal to `NaN` in `b`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Implements [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    for Awkward Arrays.
    """
    with ak._errors.OperationErrorContext(
        "ak.isclose",
        dict(
            a=a,
            b=b,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(a, b, rtol, atol, equal_nan, highlevel, behavior)


def _impl(a, b, rtol, atol, equal_nan, highlevel, behavior):
    one = ak.operations.to_layout(a)
    two = ak.operations.to_layout(b)

    def action(inputs, backend, **kwargs):
        if all(isinstance(x, ak.contents.NumpyArray) for x in inputs):
            return (
                ak.contents.NumpyArray(
                    backend.nplike.isclose(
                        inputs[0]._raw(backend.nplike),
                        inputs[1]._raw(backend.nplike),
                        rtol=rtol,
                        atol=atol,
                        equal_nan=equal_nan,
                    )
                ),
            )

    behavior = ak._util.behavior_of(a, b, behavior=behavior)
    out = ak._broadcasting.broadcast_and_apply([one, two], action, behavior)
    assert isinstance(out, tuple) and len(out) == 1

    return ak._util.wrap(out[0], behavior, highlevel)
