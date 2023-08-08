# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("without_parameters",)
import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


@high_level_function()
def without_parameters(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function returns a new array without any parameters in its
    #ak.Array.layout, on nodes of any level of depth.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    behavior = behavior_of(array, behavior=behavior)
    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)

    out = ak._do.recursively_apply(
        layout,
        (lambda layout, behavior=behavior, **kwargs: None),
        keep_parameters=False,
    )

    return wrap_layout(out, behavior, highlevel)
