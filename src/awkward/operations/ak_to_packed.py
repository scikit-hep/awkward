# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_packed(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array with the same type and values as the input, but with packed inner structures:

    - #ak.contents.NumpyArray becomes C-contiguous (if it isn't already)
    - #ak.contents.RegularArray trims unreachable content
    - #ak.contents.ListArray becomes #ak.contents.ListOffsetArray, making all list data contiguous
    - #ak.contents.ListOffsetArray starts at `offsets[0] == 0`, trimming unreachable content
    - #ak.contents.RecordArray trims unreachable contents
    - #ak.contents.IndexedArray gets projected
    - #ak.contents.IndexedOptionArray remains an #ak.contents.IndexedOptionArray (with simplified `index`)
      if it contains records, becomes #ak.contents.ByteMaskedArray otherwise
    - #ak.contents.ByteMaskedArray becomes an #ak.contents.IndexedOptionArray if it contains records,
      stays a #ak.contents.ByteMaskedArray otherwise
    - #ak.contents.BitMaskedArray becomes an #ak.contents.IndexedOptionArray if it contains records,
      stays a #ak.contents.BitMaskedArray otherwise
    - #ak.contents.UnionArray gets projected contents
    - #ak.record.Record becomes a record over a single-item #ak.contents.RecordArray

    Example:

        >>> a = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])
        >>> b = a[::-1]
        >>> b.layout
        <ListArray len='5'>
            <starts><Index dtype='int64' len='5'>
                [6 5 3 3 0]
            </Index></starts>
            <stops><Index dtype='int64' len='5'>
                [10  6  5  3  3]
            </Index></stops>
            <content><NumpyArray dtype='int64' len='10'>
                [ 1  2  3  4  5  6  7  8  9 10]
            </NumpyArray></content>
        </ListArray>

        >>> c = ak.to_packed(b)
        >>> c.layout
        <ListOffsetArray len='5'>
            <offsets><Index dtype='int64' len='6'>[ 0  4  5  7  7 10]</Index></offsets>
            <content><NumpyArray dtype='int64' len='10'>
                [ 7  8  9 10  6  4  5  1  2  3]
            </NumpyArray></content>
        </ListOffsetArray>

    Performing these operations will minimize the output size of data sent to
    #ak.to_buffers (though conversions through Arrow, #ak.to_arrow and
    #ak.to_parquet, do not need this because packing is part of that conversion).

    See also #ak.to_buffers.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_packed",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    out = layout.to_packed()
    return ak._util.wrap(out, behavior, highlevel, like=array)
