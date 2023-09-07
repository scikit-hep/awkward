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
    feather_version=2,
    # storage_options=None,
):
    """
        Args:
            df: pandas.DataFrame or pyarrow.Table
            Data to write out as Feather format.
    f
            dest: str
            Local destination path.

            compression: str, default None
            Can be one of {“zstd”, “lz4”, “uncompressed”}. The default of None uses LZ4 for V2 files if it is available, otherwise uncompressed.

            compression_level: int, default None
            Use a compression level particular to the chosen compressor. If None use the default compression level

            chunksize: int, default None
            For V2 files, the internal maximum size of Arrow RecordBatch chunks when writing the Arrow IPC file format. None means use the default, which is currently 64K

            version: int, default 2
            Feather file version. Version 2 is the current. Version 1 is the more limited legacy format


        If the `array` does not contain records at top-level, the Arrow table will consist
        of one field whose name is `""` iff. `extensionarray` is False.

        If `extensionarray` is True`, use a custom Arrow extension to store this array.
        Otherwise, generic Arrow arrays are used, and if the `array` does not
        contain records at top-level, the Arrow table will consist of one field whose
        name is `""`. See #ak.to_arrow_table for more details.

        See also #ak.to_arrow, which is used as an intermediate step.
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
        table, destination, compression, compression_level, chunksize, feather_version
    )
