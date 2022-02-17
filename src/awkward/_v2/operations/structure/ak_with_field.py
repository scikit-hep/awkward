# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable
import copy

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def with_field(base, what, where=None, highlevel=True, behavior=None):

    """
    Args:
        base: Data containing records or tuples.
        what: Data to add as a new field.
        where (None or str or non-empy iterable of str): If None, the new field
            has no name (can be accessed as an integer slot number in a
            string); If str, the name of the new field. If iterable, it is
            interpreted as a path where to add the field in a nested record.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
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

    if not (
        where is None
        or isinstance(where, str)
        or (isinstance(where, Iterable) and all(isinstance(x, str) for x in where))
    ):
        raise TypeError(
            "New fields may only be assigned by field name(s) "
            "or as a new integer slot by passing None for 'where'"
        )
    if (
        not isinstance(where, str)
        and isinstance(where, Iterable)
        and all(isinstance(x, str) for x in where)
        and len(where) > 1
    ):
        return with_field(
            base,
            with_field(
                base[where[0]],
                what,
                where=where[1:],
                highlevel=highlevel,
                behavior=behavior,
            ),
            where=where[0],
            highlevel=highlevel,
            behavior=behavior,
        )
    else:

        if not (isinstance(where, str) or where is None):
            where = where[0]

        behavior = ak._v2._util.behavior_of(base, what, behavior=behavior)
        base = ak._v2.operations.convert.to_layout(
            base, allow_record=True, allow_other=False
        )

        if len(base.fields) == 0:
            raise ValueError("no tuples or records in array; cannot add a new field")

        what = ak._v2.operations.convert.to_layout(
            what, allow_record=True, allow_other=True
        )

        keys = copy.copy(base.fields)
        if where in base.fields:
            keys.remove(where)

        if len(keys) == 0:
            # the only key was removed, so just create new Record
            out = (
                ak._v2.contents.RecordArray(
                    [what], [where], parameters=base.parameters
                ),
            )

        else:

            def action(inputs, **kwargs):
                nplike = ak.nplike.of(*inputs)
                base, what = inputs
                if isinstance(base, ak._v2.contents.RecordArray):
                    if what is None:
                        what = ak._v2.contents.IndexedOptionArray(
                            ak._v2.index.Index64(nplike.full(len(base), -1, np.int64)),
                            ak._v2.contents.EmptyArray(),
                        )
                    elif not isinstance(what, ak._v2.contents.Content):
                        what = ak._v2.contents.NumpyArray(
                            nplike.repeat(what, len(base))
                        )
                    if base.is_tuple and where is None:
                        fields = None
                    elif base.is_tuple:
                        fields = keys + [where]
                    elif where is None:
                        fields = keys + [str(len(keys))]
                    else:
                        fields = keys + [where]
                    out = ak._v2.contents.RecordArray(
                        [base[k] for k in keys] + [what],
                        fields,
                        parameters=base.parameters,
                    )
                    return (out,)
                else:
                    return None

            out = ak._v2._broadcasting.broadcast_and_apply(
                [base, what],
                action,
                behavior,
                right_broadcast=False,
            )

        assert isinstance(out, tuple) and len(out) == 1

        return ak._v2._util.wrap(out[0], behavior, highlevel)
