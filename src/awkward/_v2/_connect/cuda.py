# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

try:
    import cupy 

    error_message = None

except ModuleNotFoundError:
    cupy = None
    error_message = """to use {0}, you must install cupy:

    pip install cupy 

or

    conda install -c conda forge cupy"""

# else:
#     if ak._v2._util.parse_version(pyarrow.__version__) < ak._v2._util.parse_version(
#         "6.0.0"
#     ):
#         pyarrow = None
#         error_message = "pyarrow 6.0.0 or later required for {0}"
# 
shadow_cuda_dict = {}

def import_cupy(name):
    if cupy is None:
        raise ImportError(error_message.format(name))
    return cupy 

