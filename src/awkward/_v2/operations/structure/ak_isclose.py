# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


### FIXME: ak._v2._connect.numpy.implements needs to exist!

# @ak._v2._connect.numpy.implements("isclose")
def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, highlevel=True, behavior=None
):
    """
    Args:
        a: Array-like data (anything #ak.to_layout recognizes).
        b: Array-like data (anything #ak.to_layout recognizes).
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        equal_nan (bool): Whether to compare `NaN` as equal. If True, `NaN` in `a`
            will be considered equal to `NaN` in `b`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Implements [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
    for Awkward Arrays.
    """
    one = ak._v2.operations.convert.to_layout(a)
    two = ak._v2.operations.convert.to_layout(b)

    def action(inputs, nplike, **kwargs):
        if all(isinstance(x, ak._v2.contents.NumpyArray) for x in inputs):
            return (
                ak._v2.contents.NumpyArray(
                    nplike.isclose(
                        inputs[0].to(nplike),
                        inputs[1].to(nplike),
                        rtol=rtol,
                        atol=atol,
                        equal_nan=equal_nan,
                    )
                ),
            )

    behavior = ak._v2._util.behavior_of(a, b, behavior=behavior)
    out = ak._v2._broadcasting.broadcast_and_apply([one, two], action, behavior)
    assert isinstance(out, tuple) and len(out) == 1

    return ak._v2._util.wrap(out[0], behavior, highlevel)
