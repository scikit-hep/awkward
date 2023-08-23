# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numba
from numba.core.errors import NumbaTypeError

import awkward as ak
from awkward._backends.cupy import CupyBackend

########## ArrayView Arguments Handler for CUDA JIT


class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        if isinstance(val, ak.Array):
            if isinstance(val.layout.backend, CupyBackend):
                # Use uint64 for pos, start, stop, the array pointers values, and the pylookup value
                tys = numba.types.UniTuple(numba.types.uint64, 5)

                start = val._numbaview.start
                stop = val._numbaview.stop
                pos = val._numbaview.pos
                arrayptrs = val._numbaview.lookup.arrayptrs.data.ptr
                pylookup = 0

                return tys, (pos, start, stop, arrayptrs, pylookup)
            else:
                raise NumbaTypeError(
                    '`ak.to_backend` should be called with `backend="cuda"` to put '
                    "the array on the GPU before using it: "
                    'ak.to_backend(array, backend="cuda")'
                )

        else:
            return ty, val
