# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._regularize import is_non_string_like_sequence

__all__ = ("with_field",)

np = NumpyMetadata.instance()


@high_level_function()
def with_field(array, what, where=None, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        what: Array-like data (anything #ak.to_layout recognizes) to add as a new field.
        where (None or str or non-empy sequence of str): If None, the new field
            has no name (can be accessed as an integer slot number in a
            string); If str, the name of the new field. If a sequence, it is
            interpreted as a path where to add the field in a nested record.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new field attached. This function does not
    change the array in-place.

    See #ak.Array.__setitem__ and #ak.Record.__setitem__ for a variant that
    changes the high-level object in-place. (These methods internally use
    #ak.with_field, so performance is not a factor in choosing one over the
    other.)
    """
    # Dispatch
    yield array, what

    # Implementation
    return _impl(array, what, where, highlevel, behavior, attrs)


def _impl(base, what, where, highlevel, behavior, attrs):
    if not (
        where is None
        or isinstance(where, str)
        or (
            is_non_string_like_sequence(where)
            and all(isinstance(x, str) for x in where)
        )
    ):
        raise TypeError(
            "New fields may only be assigned by field name(s) "
            "or as a new integer slot by passing None for 'where'"
        )

    if is_non_string_like_sequence(where) and len(where) > 1:
        return _impl(
            base,
            _impl(base[where[0]], what, where[1:], highlevel, behavior, attrs),
            where[0],
            highlevel,
            behavior,
            attrs,
        )
    else:
        # If we have an iterable here, pull out the only ti
        if is_non_string_like_sequence(where):
            where = where[0]

        with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
            base, what = ensure_same_backend(
                ctx.unwrap(base, allow_record=True, primitive_policy="error"),
                ctx.unwrap(
                    what,
                    allow_record=True,
                    allow_unknown=False,
                    none_policy="pass-through",
                    primitive_policy="pass-through",
                    string_policy="promote",
                ),
            )

        keys = copy.copy(base.fields)
        if where in base.fields:
            keys.remove(where)

        def purelist_is_record(layout):
            result = False

            def action_is_record(input, **kwargs):
                nonlocal result

                if input.is_record:
                    result = True
                    return input
                elif input.is_union:
                    result = all(purelist_is_record(x) for x in input.contents)
                    return input
                else:
                    return None

            ak._do.recursively_apply(layout, action_is_record, return_array=False)
            return result

        if not purelist_is_record(base):
            raise ValueError("no tuples or records in array; cannot add a new field")

        def action(inputs, **kwargs):
            base, what = inputs
            backend = base.backend

            if isinstance(base, ak.contents.RecordArray):
                if what is None:
                    what = ak.contents.IndexedOptionArray(
                        ak.index.Index64(
                            backend.index_nplike.full(base.length, -1, dtype=np.int64),
                            nplike=backend.index_nplike,
                        ),
                        ak.contents.EmptyArray(),
                    )
                elif not isinstance(what, ak.contents.Content):
                    what = ak.contents.NumpyArray(
                        backend.nplike.repeat(
                            backend.nplike.asarray(what),
                            backend.nplike.shape_item_as_index(base.length),
                        )
                    )
                if base.is_tuple:
                    # Preserve tuple-ness
                    if where is None:
                        fields = None
                    # Otherwise the tuple becomes a record
                    else:
                        fields = [*keys, where]
                # Records with `where=None` will create a tuple-like key
                elif where is None:
                    fields = [*keys, str(len(keys))]
                else:
                    fields = [*keys, where]
                out = ak.contents.RecordArray(
                    [base[k] for k in keys] + [what],
                    fields,
                    parameters=base.parameters,
                )
                return (out,)
            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            [base, what], action, right_broadcast=False
        )

        assert isinstance(out, tuple) and len(out) == 1

        return ctx.wrap(out[0], highlevel=highlevel)
