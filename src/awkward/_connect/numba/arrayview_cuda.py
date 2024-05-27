# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numba
from numba.core.errors import NumbaTypeError

import awkward as ak
from awkward._backends.cupy import CupyBackend

########## ArrayView Arguments Handler for CUDA JIT


class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        if isinstance(val, ak.Array):
            if isinstance(val.layout.backend, CupyBackend):
                if ty is not val.numba_type:
                    raise NumbaTypeError(
                        f"the array type: {val.numba_type} does not match "
                        f"the kernel signature type: {ty}"
                    )

                # Use uint64 for pos, start, stop, the array pointers values, and the pylookup value
                tys = numba.types.UniTuple(numba.types.uint64, 5)

                view = val._numbaview
                assert view is not None

                start = view.start
                stop = view.stop
                pos = view.pos
                arrayptrs = view.lookup.arrayptrs.data.ptr
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
