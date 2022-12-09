# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy
from collections.abc import Sequence

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def with_field(array, what, where=None, *, highlevel=True, behavior=None):
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

    Returns an #ak.Array or #ak.Record (or low-level equivalent, if
    `highlevel=False`) with a new field attached. This function does not
    change the array in-place.

    See #ak.Array.__setitem__ and #ak.Record.__setitem__ for a variant that
    changes the high-level object in-place. (These methods internally use
    #ak.with_field, so performance is not a factor in choosing one over the
    other.)
    """
    with ak._errors.OperationErrorContext(
        "ak.with_field",
        dict(
            array=array, what=what, where=where, highlevel=highlevel, behavior=behavior
        ),
    ):
        return _impl(array, what, where, highlevel, behavior)


def _impl(base, what, where, highlevel, behavior):
    if not (
        where is None
        or isinstance(where, str)
        or (
            ak._util.is_non_string_iterable(where)
            and isinstance(where, Sequence)
            and all(isinstance(x, str) for x in where)
        )
    ):
        raise ak._errors.wrap_error(
            TypeError(
                "New fields may only be assigned by field name(s) "
                "or as a new integer slot by passing None for 'where'"
            )
        )

    if ak._util.is_non_string_iterable(where) and len(where) > 1:
        return _impl(
            base,
            _impl(
                base[where[0]],
                what,
                where[1:],
                highlevel,
                behavior,
            ),
            where[0],
            highlevel,
            behavior,
        )
    else:
        # If we have an iterable here, pull out the only ti
        if ak._util.is_non_string_iterable(where):
            where = where[0]

        behavior = ak._util.behavior_of(base, what, behavior=behavior)
        base = ak.operations.to_layout(base, allow_record=True, allow_other=False)

        if len(base.fields) == 0:
            raise ak._errors.wrap_error(
                ValueError("no tuples or records in array; cannot add a new field")
            )

        what = ak.operations.to_layout(what, allow_record=True, allow_other=True)

        keys = copy.copy(base.fields)
        if where in base.fields:
            keys.remove(where)

        def action(inputs, **kwargs):
            base, what = inputs
            backend = base.backend

            if isinstance(base, ak.contents.RecordArray):
                if what is None:
                    what = ak.contents.IndexedOptionArray(
                        ak.index.Index64(
                            backend.index_nplike.full(len(base), -1, np.int64),
                            nplike=backend.index_nplike,
                        ),
                        ak.contents.EmptyArray(),
                    )
                elif not isinstance(what, ak.contents.Content):
                    what = ak.contents.NumpyArray(
                        backend.nplike.repeat(what, len(base))
                    )
                if base.is_tuple:
                    # Preserve tuple-ness
                    if where is None:
                        fields = None
                    # Otherwise the tuple becomes a record
                    else:
                        fields = keys + [where]
                # Records with `where=None` will create a tuple-like key
                elif where is None:
                    fields = keys + [str(len(keys))]
                else:
                    fields = keys + [where]
                out = ak.contents.RecordArray(
                    [base[k] for k in keys] + [what],
                    fields,
                    parameters=base.parameters,
                )
                return (out,)
            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            [base, what],
            action,
            behavior,
            right_broadcast=False,
        )

        assert isinstance(out, tuple) and len(out) == 1

        return ak._util.wrap(out[0], behavior, highlevel)
