# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def is_tuple(array):
    """
    Args:
        array (#ak.Array, #ak.Record, #ak.layout.Content, #ak.layout.Record, #ak.ArrayBuilder, #ak.layout.ArrayBuilder):
            Array or record to check.

    If `array` is a record, this returns True if the record is a tuple.
    If `array` is an array, this returns True if the outermost record is a tuple.
    """
    with ak._util.OperationErrorContext(
        "ak.is_tuple",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    layout = ak.to_layout(array, allow_record=True)

    return layout.is_tuple
