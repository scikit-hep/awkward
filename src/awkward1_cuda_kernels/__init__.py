# BSD 3-Clause License; see https://github.com/scikit-hep/awkward1/blob/master/LICENSE

from __future__ import absolute_import

import os.path
import glob

directory, filename = os.path.split(__file__)

shared_library_path = None
for filename in glob.glob(os.path.join(directory, "*awkward-cuda-kernels.*")):
    shared_library_path = filename

del os
del glob
del directory
del filename

