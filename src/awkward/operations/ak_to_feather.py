# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import os

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_feather",)

metadata = NumpyMetadata.instance()


@high_level_function()
def to_feather(
    array,
    destination,
    *,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
    emptyarray_to=None,
    categorical_as_dictionary=False,
    extensionarray=True,
    count_nulls=True,
    compression="zstd",
    compression_level=None,
    chunksize=None,
    feather_version=2,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        destination (str): Local destination path, passed to
            [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.types.ListType maps to Arrow `ListType`,
            signed 64-bit #ak.types.ListType maps to Arrow `LargeListType`,
            and unsigned 32-bit #ak.types.ListType picks whichever Arrow type its
            values fit into.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
        emptyarray_to (None or dtype): If None, #ak.types.UnknownType maps to Arrow's
            null type; otherwise, it is converted a given numeric dtype.
        categorical_as_dictionary (bool): If True, #ak.contents.IndexedArray and
            #ak.contents.IndexedOptionArray labeled with `__array__ = "categorical"`
            are mapped to Arrow `DictionaryArray`; otherwise, the projection is
            evaluated before conversion (always the case without
            `__array__ = "categorical"`).
        extensionarray (bool): If True, this function returns extended Arrow arrays
            (at all levels of nesting), which preserve metadata so that Awkward \u2192
            Arrow \u2192 Awkward preserves the array's #ak.types.Type (though not
            the #ak.forms.Form). If False, this function returns generic Arrow arrays
            that might be needed for third-party tools that don't recognize Arrow's
            extensions. Even with `extensionarray=False`, the values produced by
            Arrow's `to_pylist` method are the same as the values produced by Awkward's
            #ak.to_list.
        count_nulls (bool): If True, count the number of missing values at each level
            and include these in the resulting Arrow array, which makes some downstream
            applications faster. If False, skip the up-front cost of counting them.
        compression (None or str): Can be one of {“zstd”, “lz4”, “uncompressed”}. The
            default of None uses LZ4 for `feather_version=2` files if it is available, otherwise
            uncompressed. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        compression_level (None or int): Use a compression level particular to the chosen
            compressor. If None use the default compression level. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        chunksize (None or int): For `feather_version=2` files, this is the internal maximum size of Arrow RecordBatch
            chunks when writing the Arrow IPC file format. None means use the
            default, which is currently 64K. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        feather_version (int): Feather file version, passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
            Version 2 is the current. Version 1 is the more limited legacy format. If not
            provided, version 2 is used.

    Writes an Awkward Array to a Feather file (through pyarrow).

        >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        >>> ak.to_feather(array, "filename.feather")

    If the `array` does not contain records at top-level, the Arrow table will
    consist of one field whose name is `""` iff. `extensionarray` is False.

    If `extensionarray` is True`, use a custom Arrow extension to store this array.
    Otherwise, generic Arrow arrays are used, and if the `array` does not
    contain records at top-level, the Arrow table will consist of one field whose
    name is `""`. See #ak.to_arrow_table for more details.

    See also #ak.from_feather.
    """

    # Dispatch
    yield (array,)

    return _impl(
        array,
        list_to32,
        string_to32,
        bytestring_to32,
        emptyarray_to,
        categorical_as_dictionary,
        extensionarray,
        count_nulls,
        destination,
        compression,
        compression_level,
        chunksize,
        feather_version,
    )


def _impl(
    array,
    list_to32,
    string_to32,
    bytestring_to32,
    emptyarray_to,
    categorical_as_dictionary,
    extensionarray,
    count_nulls,
    destination,
    compression,
    compression_level,
    chunksize,
    feather_version,
):

    layout = ak.operations.ak_to_layout._impl(
        array,
        allow_record=True,
        allow_unknown=False,
        none_policy="error",
        regulararray=True,
        use_from_iter=True,
        primitive_policy="error",
        string_policy="as-characters",
    )

    table = ak.operations.ak_to_arrow_table._impl(
        layout,
        list_to32,
        string_to32,
        bytestring_to32,
        emptyarray_to,
        categorical_as_dictionary,
        extensionarray,
        count_nulls,
    )

    try:
        destination = os.fsdecode(destination)
    except TypeError:
        raise TypeError(
            f"'destination' argument of 'ak.to_feather' must be a path-like, "
            f"not {type(destination).__name__} "
            f"('array' argument is first; 'destination' second)"
        ) from None

    # Normalize `compression` the same way for both backends below:
    # True -> "zstd", False/None -> Python None ("no compression").
    # NOTE: intentionally NOT the string "none" here - that was never a
    # valid pyarrow value and raised ValueError in the pre-existing code
    # whenever a caller passed compression=None or compression=False
    # explicitly (both hit the same underlying default path in practice,
    # since the public default is compression="zstd").
    if compression is True:
        compression = "zstd"
    elif compression is False or compression is None:
        compression = None

    if feather_version < 2:
        # Feather V2 (and, presumably, any future version >= 2) is the
        # Arrow IPC file format on disk. Version 1 predates the IPC format
        # entirely and cannot be written via pyarrow.ipc at all - pyarrow
        # offers no non-deprecated way to produce a V1 file, so we fall
        # back to the deprecated pyarrow.feather API here, scoped narrowly
        # to this one legacy case, and let pyarrow validate `feather_version`
        # itself (e.g. reject negative or zero values).
        import warnings

        import pyarrow.feather

        compression_arg = "none" if compression is None else compression

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pyarrow.feather.write_feather is deprecated.*",
                category=FutureWarning,
            )
            pyarrow.feather.write_feather(
                table,
                destination,
                compression=compression_arg,
                compression_level=compression_level,
                chunksize=chunksize,
                version=feather_version,
            )
        return

    # feather_version >= 2: write via the non-deprecated pyarrow.ipc API.
    # This produces the same on-disk format as the old
    # pyarrow.feather.write_feather call did for V2 - a different
    # (non-deprecated) API for writing the same bytes, not a different
    # format.
    import pyarrow
    import pyarrow.ipc

    if compression is None:
        codec = None
    elif compression_level is None:
        codec = compression
    else:
        codec = pyarrow.Codec(compression, compression_level)

    options = pyarrow.ipc.IpcWriteOptions(
        compression=codec,
    )

    with pyarrow.OSFile(destination, "wb") as sink:
        with pyarrow.ipc.new_file(
            sink,
            table.schema,
            options=options,
        ) as writer:
            # `max_chunksize` is the pyarrow.ipc equivalent of the old
            # `chunksize` parameter - same meaning (max rows per
            # RecordBatch chunk), renamed keyword in the new API.
            writer.write_table(
                table,
                max_chunksize=chunksize,
            )
