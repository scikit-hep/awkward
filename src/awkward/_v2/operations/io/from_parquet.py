# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_parquet(  # this will be going through Awkward-Dask
    source,
    columns=None,
    row_groups=None,
    use_threads=True,
    include_partition_columns=True,
    lazy=False,
    lazy_cache="new",
    lazy_cache_key=None,
    highlevel=True,
    behavior=None,
    **options  # NOTE: a comma after **options breaks Python 2
):
    raise NotImplementedError


#     """
#     Args:
#         source (str, Path, file-like object, pyarrow.NativeFile): Where to
#             get the Parquet file. If `source` is the name of a local directory
#             (str or Path), then it is interpreted as a partitioned Parquet dataset.
#         columns (None or list of str): If None, read all columns; otherwise,
#             read a specified set of columns.
#         row_groups (None, int, or list of int): If None, read all row groups;
#             otherwise, read a single or list of row groups.
#         use_threads (bool): Passed to the pyarrow.parquet.ParquetFile.read
#             functions; if True, do multithreaded reading.
#         include_partition_columns (bool): If True and `source` is a partitioned
#             Parquet dataset with subdirectory names defining partition names
#             and values, include those special columns in the output.
#         lazy (bool): If True, read columns in row groups on demand (as
#             #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
#             if the file has more than one row group); if False, read all
#             requested data immediately.
#         lazy_cache (None, "new", or MutableMapping): If lazy, pass this
#             cache to the VirtualArrays. If "new", a new dict (keep-forever cache)
#             is created. If None, no cache is used.
#         lazy_cache_key (None or str): If lazy, pass this cache_key to the
#             VirtualArrays. If None, a process-unique string is constructed.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.
#         options: All other options are passed to pyarrow.parquet.ParquetFile.

#     Reads a Parquet file into an Awkward Array (through pyarrow).

#         >>> ak.from_parquet("array1.parquet")
#         <Array [[1, 2, 3], [], ... [], [6, 7, 8, 9]] type='6 * var * ?int64'>

#     See also #ak.from_arrow, which is used as an intermediate step.
#     See also #ak.to_parquet.
#     """
