# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.virtual import VirtualNDArray
from awkward._typing import Mapping

__all__ = ("nan_to_none",)

np = NumpyMetadata.instance()


@high_level_function()
def nan_to_none(
    array,
    *,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping | None = None,
):
    """Converts NaN ("not a number") into None, i.e. missing values with option-type.

    On CPU, a complex value is missing if either component is NaN.

    For complex values, either a real or imaginary NaN makes the entire value
    missing. This rule is the same for NumPy, CuPy, JAX and typetracer layouts.

    See also #ak.nan_to_num to convert NaN or infinity to specified values.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns:
        An array with NaN ("not a number") converted to None, i.e. missing
        values with option-type.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(
    array,
    highlevel: bool,
    behavior: Mapping | None,
    attrs: Mapping | None,
    *,
    optimize_all_valid: bool = False,
):
    def supports_nan_mask(layout):
        return layout.is_numpy and (
            np.issubdtype(layout.dtype, np.floating)
            or np.issubdtype(layout.dtype, np.complexfloating)
        )

    def action(layout, continuation, backend, **kwargs):
        if supports_nan_mask(layout):
            mask = backend.nplike.isnan(layout.data)
            if (
                optimize_all_valid
                and backend is NumpyBackend.instance()
                and layout.dtype.kind == "c"
                and not isinstance(mask, VirtualNDArray)
                and mask.size != 0
                and not mask.any()
            ):
                # Keep option semantics while allowing reductions to skip
                # mask projection and its carry of every valid value.
                return ak.contents.UnmaskedArray(layout)
            return ak.contents.ByteMaskedArray(
                ak.index.Index8(mask, nplike=backend.nplike),
                layout,
                valid_when=False,
            )

        elif (layout.is_option or layout.is_indexed) and (
            supports_nan_mask(layout.content)
        ):
            return continuation()

        else:
            return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)
