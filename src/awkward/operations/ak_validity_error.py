# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def validity_error(array, *, exception=False):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        exception (bool): If True, validity errors raise exceptions.

    Returns an empty string if there are no errors and a str containing the error message
    if there are.

    Checks for errors in the structure of the array, such as indexes that run
    beyond the length of a node's `content`, etc. Either an error is raised or
    a string describing the error is returned.

    See also #ak.is_valid.
    """
    with ak._errors.OperationErrorContext(
        "ak.validity_error",
        dict(array=array, exception=exception),
    ):
        return _impl(array, exception)


def _impl(array, exception):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    out = ak._do.validity_error(layout, path="highlevel")

    if out not in (None, "") and exception:
        raise ak._errors.wrap_error(ValueError(out))
    else:
        return out
