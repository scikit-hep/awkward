# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import awkward1.layout
import awkward1._numba

__version__ = awkward1.layout.__version__

if awkward1._numba.installed:
    dummy1 = awkward1._numba.cpu.kernels.dummy1

dummy3 = layout.dummy3
