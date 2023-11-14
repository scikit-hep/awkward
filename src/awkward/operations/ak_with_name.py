# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("with_name",)

np = NumpyMetadata.instance()


@high_level_function()
def with_name(array, name, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        name (str or None): Name to give to the records or tuples; this assigns
            the `"__record__"` parameter. If None, any existing name is unset.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
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
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, name, highlevel, behavior, attrs)


def _impl(array, name, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=True, primitive_policy="error")

    def action(layout, **ignore):
        if isinstance(layout, ak.contents.RecordArray):
            return ak.contents.RecordArray(
                layout._contents,
                layout._fields,
                layout._length,
                parameters={**layout.parameters, "__record__": name},
            )
        else:
            return None

    out = ak._do.recursively_apply(layout, action)

    return ctx.wrap(out, highlevel=highlevel)
