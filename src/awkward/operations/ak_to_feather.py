# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_feather",)
from os import fsdecode

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpylike import NumpyMetadata

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
    version=2,
):
    """
    Args:
        array (pandas.DataFrame, or pyarrow.Table): Data to write out as Feather format,
            passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        destination (str): Local destination path, passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
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
        compression (str): Can be one of {“zstd”, “lz4”, “uncompressed”}. The
            default of None uses LZ4 for V2 files if it is available, otherwise
            uncompressed. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        compression_level (int): Use a compression level particular to the chosen
            compressor. If None use the default compression level. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        chunksize (int): For V2 files, the internal maximum size of Arrow RecordBatch
            chunks when writing the Arrow IPC file format. None means use the
            default, which is currently 64K. Passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
        version (int): Feather file version, passed to [pyarrow.feather.write_feather](https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html#pyarrow.feather.write_feather).
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
        version,
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
    version,
):
    import pyarrow.feather

    layout = ak.operations.ak_to_layout._impl(
        array, allow_record=True, allow_other=False, regulararray=True
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

    if compression is True:
        compression = "zstd"
    elif compression is False or compression is None:
        compression = "none"

    try:
        destination = fsdecode(destination)
    except TypeError:
        raise TypeError(
            f"'destination' argument of 'ak.to_feather' must be a path-like, not {type(destination).__name__} ('array' argument is first; 'destination' second)"
        ) from None

    pyarrow.feather.write_feather(
        table, destination, compression, compression_level, chunksize, version
    )
