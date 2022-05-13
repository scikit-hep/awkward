# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def with_parameter(array, parameter, value, highlevel=True, behavior=None):
    """
    Args:
        array: Data convertible into an Awkward Array.
        parameter (str): Name of the parameter to set on that array.
        value (JSON): Value of the parameter to set on that array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    This function returns a new array with a parameter set on the outermost
    node of its #ak.Array.layout.

    Note that a "new array" is a lightweight shallow copy, not a duplication
    of large data buffers.

    You can also remove a single parameter with this function, since setting
    a parameter to None is equivalent to removing it.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.with_parameter",
        dict(
            array=array,
            parameter=parameter,
            value=value,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, parameter, value, highlevel, behavior)


def _impl(array, parameter, value, highlevel, behavior):
    behavior = ak._v2._util.behavior_of(array, behavior=behavior)
    layout = ak._v2.operations.to_layout(array, allow_record=True, allow_other=False)

    out = layout.with_parameter(parameter, value)

    return ak._v2._util.wrap(
        out, ak._v2._util.behavior_of(array, behavior=behavior), highlevel
    )
