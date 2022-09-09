# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: keep this file, but modify it to only get objects that exist!

# NumPy-like alternatives
import awkward.nplike

# shims for C++ (now everything is compiled into one 'awkward._ext' module)
import awkward.layout

# internal
import awkward._v2
import awkward._cpu_kernels
import awkward._libawkward

# version
__version__ = awkward._ext.__version__

# call C++ startup function
awkward._ext.startup()

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numpy",)]


def __dir__():
    return __all__
