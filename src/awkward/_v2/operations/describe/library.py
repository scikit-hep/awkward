# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def library(*arrays):  # note: convert.py's 'kernels'
    raise NotImplementedError


#     """
#     Returns the names of the kernels library used by `arrays`. May be

#        * `"cpu"` for `libawkward-cpu-kernels.so`;
#        * `"cuda"` for `libawkward-cuda-kernels.so`;
#        * `"mixed"` if any of the arrays have different labels within their
#          structure or any arrays have different labels from each other;
#        * None if the objects are not Awkward, NumPy, or CuPy arrays (e.g.
#          Python numbers, booleans, strings).

#     Mixed arrays can't be used in any operations, and two arrays on different
#     devices can't be used in the same operation.

#     To use `"cuda"`, the package
#     [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
#     be installed, either by

#         pip install awkward-cuda-kernels

#     or as an optional dependency with

#         pip install awkward[cuda] --upgrade

#     It is only available for Linux as a binary wheel, and only supports Nvidia
#     GPUs (it is written in CUDA).

#     See #ak.to_kernels.
#     """
#     libs = set()
#     for array in arrays:
#         layout = ak._v2.operations.convert.to_layout(
#             array,
#             allow_record=True,
#             allow_other=True,
#         )

#         if isinstance(
#             layout, (ak._v2.contents.Content, ak._v2.record.Record, ak.partition.PartitionedArray)   # NO PARTITIONED ARRAY
#         ):
#             libs.add(layout.kernels)

#         elif isinstance(layout, ak.nplike.numpy.ndarray):
#             libs.add("cpu")

#         elif type(layout).__module__.startswith("cupy."):
#             libs.add("cuda")

#     if libs == set():
#         return None
#     elif libs == set(["cpu"]):
#         return "cpu"
#     elif libs == set(["cuda"]):
#         return "cuda"
#     else:
#         return "mixed"
