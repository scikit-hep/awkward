# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def with_name(array, name, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        name (str or None): Name to give to the records or tuples; this assigns
            the `"__record__"` parameter. If None, any existing name is unset.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new name. This function does not change the
    array in-place. If the new name is None, then an array without a name is
    returned.

    The records or tuples may be nested within multiple levels of nested lists.
    If records are nested within records, only the outermost are affected.

    Setting the `"__record__"` parameter makes it possible to add behaviors
    to the data; see #ak.Array and #ak.behavior for a more complete
    description.
    """
    with ak._errors.OperationErrorContext(
        "ak.with_name",
        dict(array=array, name=name, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, name, highlevel, behavior)


def _impl(array, name, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.operations.to_layout(array)

    def action(layout, **ignore):
        if isinstance(layout, ak.contents.RecordArray):
            parameters = dict(layout.parameters)
            parameters["__record__"] = name
            return ak.contents.RecordArray(
                layout.contents, layout.fields, len(layout), parameters=parameters
            )
        else:
            return None

    out = ak._do.recursively_apply(layout, action, behavior)

    def action2(layout, **ignore):
        if layout.is_union:
            return ak.contents.UnionArray.simplified(
                layout._tags,
                layout._index,
                layout._contents,
                parameters=layout._parameters,
            )
        else:
            return None

    out2 = ak._do.recursively_apply(out, action2, behavior)

    return ak._util.wrap(out2, behavior, highlevel)
