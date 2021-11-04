# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_library(
    array, kernels, highlevel=True, behavior=None
):  # note: convert.py's 'to_kernels'
    raise NotImplementedError


#     """
#     Args:
#         array: Data to convert to a specified `kernels` set.
#         kernels (`"cpu"` or `"cuda"`): If `"cpu"`, the array structure is
#             recursively copied (if need be) to main memory for use with
#             the default `libawkward-cpu-kernels.so`; if `"cuda"`, the
#             structure is copied to the GPU(s) for use with
#             `libawkward-cuda-kernels.so`.
#         highlevel (bool): If True, return an #ak.Array; otherwise, return
#             a low-level #ak.layout.Content subclass.
#         behavior (None or dict): Custom #ak.behavior for the output array, if
#             high-level.

#     Converts an array from `"cpu"`, `"cuda"`, or `"mixed"` kernels to `"cpu"`
#     or `"cuda"`.

#     An array is `"mixed"` if some components are set to use `"cpu"` kernels and
#     others are set to use `"cuda"` kernels. Mixed arrays can't be used in any
#     operations, and two arrays set to different kernels can't be used in the
#     same operation.

#     Any components that are already in the desired kernels library are viewed,
#     rather than copied, so this operation can be an inexpensive way to ensure
#     that an array is ready for a particular library.

#     To use `"cuda"`, the package
#     [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
#     be installed, either by

#         pip install awkward-cuda-kernels

#     or as an optional dependency with

#         pip install awkward[cuda] --upgrade

#     It is only available for Linux as a binary wheel, and only supports Nvidia
#     GPUs (it is written in CUDA).

#     See #ak.kernels.
#     """
#     arr = ak.to_layout(array)
#     out = arr.copy_to(kernels)

#     return ak._v2._util.maybe_wrap_like(out, array, behavior, highlevel)
