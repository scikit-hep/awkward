# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Sequence

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def without_field(array, where, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        where (str or non-empy sequence of str): If str, the name of the field
            to be removed. If a sequence, it is interpreted as a path where to
            remove the field in a nested record.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with an existing field removed. This function does not
    change the array in-place.

    See #ak.Array.__delitem__ and #ak.Record.__delitem__ for a variant that
    changes the high-level object in-place. (These methods internally use
    #ak.without_field, so performance is not a factor in choosing one over the
    other.)
    """
    with ak._errors.OperationErrorContext(
        "ak.without_field",
        dict(array=array, where=where, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, where, highlevel, behavior)


def _impl(base, where, highlevel, behavior):
    if isinstance(where, str):
        where = [where]
    elif not (isinstance(where, Sequence) and all(isinstance(x, str) for x in where)):
        raise ak._errors.wrap_error(
            TypeError(
                "Field names must be given as a single string, or a sequence of strings"
            )
        )

    behavior = ak._util.behavior_of(base, behavior=behavior)
    base = ak.operations.to_layout(base, allow_record=True, allow_other=False)

    def action(layout, depth_context, **kwargs):
        if isinstance(layout, ak.contents.RecordArray):
            field, *next_where = depth_context["where"]

            # For parent record arrays, we don't change the fields, just
            # modify the contents
            i_field = layout.field_to_index(field)
            if len(next_where):
                next_contents = []
                for i, content in enumerate(layout.contents):
                    if i == i_field:
                        # Visit this content to remove the next item in `where`
                        next_content = ak._do.recursively_apply(
                            content,
                            action,
                            behavior,
                            depth_context={"where": next_where},
                        )
                        next_contents.append(next_content)
                    else:
                        next_contents.append(content)
                return layout.copy(contents=next_contents)
            # If we're at the final layout
            else:
                next_contents = [
                    c for i, c in enumerate(layout.contents) if i != i_field
                ]

                if layout.is_tuple:
                    next_fields = None
                else:
                    next_fields = [
                        f for i, f in enumerate(layout.fields) if i != i_field
                    ]

                return layout.copy(contents=next_contents, fields=next_fields)
        else:
            return None

    out = ak._do.recursively_apply(
        base, action, behavior, depth_context={"where": where}
    )
    return ak._util.wrap(out, behavior, highlevel)
