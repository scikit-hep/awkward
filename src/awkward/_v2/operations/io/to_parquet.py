# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_parquet(  # this will be going through Awkward-Dask
    array,
    where,
    explode_records=False,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
    **options  # NOTE: a comma after **options breaks Python 2
):
    raise NotImplementedError


#     """
#     Args:
#         array: Data to write to a Parquet file.
#         where (str, Path, file-like object): Where to write the Parquet file.
#         explode_records (bool): If True, lists of records are written as
#             records of lists, so that nested fields become top-level fields
#             (which can be zipped when read back).
#         list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
#             if they're small enough, even if it means an extra conversion. Otherwise,
#             signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
#             all others map to Arrow `LargeListType`.
#         string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
#         bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
#         options: All other options are passed to pyarrow.parquet.ParquetWriter.
#             In particular, if no `schema` is given, a schema is derived from
#             the array type.

#     Writes an Awkward Array to a Parquet file (through pyarrow).

#         >>> array1 = ak.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
#         >>> ak.to_parquet(array1, "array1.parquet")

#     If the `array` does not contain records at top-level, the Arrow table will consist
#     of one field whose name is `""`.

#     Parquet files can maintain the distinction between "option-type but no elements are
#     missing" and "not option-type" at all levels, including the top level. However,
#     there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
#     Be aware of these type distinctions when passing data through Arrow or Parquet.

#     To make a partitioned Parquet dataset, use this function to write each Parquet
#     file to a directory (as separate invocations, probably in parallel with multiple
#     processes), then give them common metadata by calling `ak.to_parquet.dataset`.

#         >>> ak.to_parquet(array1, "directory-name/file1.parquet")
#         >>> ak.to_parquet(array2, "directory-name/file2.parquet")
#         >>> ak.to_parquet(array3, "directory-name/file3.parquet")
#         >>> ak.to_parquet.dataset("directory-name")

#     Then all of the flies in the collection can be addressed as one array. For example,

#         >>> dataset = ak.from_parquet("directory_name", lazy=True)

#     (If it is large, you will likely want to load it lazily.)

#     See also #ak.to_arrow, which is used as an intermediate step.
#     See also #ak.from_parquet.
#     """
