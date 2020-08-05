# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1._ext
import awkward1._util

def copy_to(array, ptr_lib, highlevel=True, behavior=None):
    if (isinstance(array, awkward1.layout.Content) or
        isinstance(array, awkward1.layout.Index8) or
        isinstance(array, awkward1.layout.IndexU8) or
        isinstance(array, awkward1.layout.Index32) or
        isinstance(array, awkward1.layout.IndexU32) or
        isinstance(array, awkward1.layout.Index64)):
        return array.copy_to(ptr_lib)

    arr = awkward1.to_layout(array)
    if highlevel:
        return awkward1._util.wrap(arr.copy_to(ptr_lib), behavior)
    else:
        return arr.copy_to(ptr_lib)

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("awkward1")]
