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
    with ak._v2._util.OperationErrorContext(
        "ak._v2.is_tuple",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    layout = ak._v2.to_layout(array, allow_record=True)

    if isinstance(layout, ak._v2.record.Record):
        return layout.is_tuple

    def visitor(form):
        if form.is_RecordType:
            return form.is_tuple
        elif form.is_ListType or form.is_OptionType or form.is_IndexedType:
            return visitor(form.content)
        elif form.is_UnionType:
            return all(visitor(x) for x in form.contents)
        elif form.is_NumpyType or form.is_UnknownType:
            return False
        else:
            raise ak._v2._util.error(
                ValueError(f"Unexpected layout type {type(layout).__name__}")
            )

    return visitor(layout.form)
