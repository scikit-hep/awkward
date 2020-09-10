# BSD 3-Clause License; see https://github.com/scikit-hep/awkward1/blob/master/LICENSE

from __future__ import absolute_import

import platform
import pkg_resources

# awkward1-cuda-kernels is only supported on Linux, but let's leave the placeholder.
if platform.system() == "Windows":
    shared_library_name = "awkward-cuda-kernels.dll"
elif platform.system() == "Darwin":
    shared_library_name = "libawkward-cuda-kernels.dylib"
else:
    shared_library_name = "libawkward-cuda-kernels.so"

shared_library_path = pkg_resources.resource_filename(
    "awkward1_cuda_kernels", shared_library_name
)

del platform
del pkg_resources
