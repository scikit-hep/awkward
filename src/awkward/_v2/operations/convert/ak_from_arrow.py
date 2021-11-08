# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_arrow(array, highlevel=True, behavior=None):
    """
    Args:
        array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`,
            or `pyarrow.Table`): Apache Arrow array to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
    """
    import awkward._v2._connect.pyarrow

    out = awkward._v2._connect.pyarrow.handle_arrow(array, pass_empty_field=True)
    return ak._v2._util.wrap(out, behavior, highlevel)
