# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import numba.cuda as nb_cuda
from numba import types

import awkward as ak

########## ArrayView Arguments Handler for CUDA JIT


class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        if isinstance(val, ak.Array):

            if isinstance(val.layout.backend, ak._backends.CupyBackend):

                # Use uint64 for pos, start, stop, the array pointers values, and the pylookup value
                tys = types.UniTuple(types.uint64, 5)

                nb_cuda.as_cuda_array(val.layout.data)

                start = val._numbaview.start
                stop = val._numbaview.stop
                pos = val._numbaview.pos
                arrayptrs = val._numbaview.lookup.arrayptrs.data.ptr
                pylookup = 0

                return tys, (pos, start, stop, arrayptrs, pylookup)
            else:
                raise ak._errors.wrap_error(
                    NotImplementedError(
                        f"{repr(val.layout.nplike)} is not implemented for CUDA. Please transfer the array to CUDA backend to "
                        "continue the operation."
                    )
                )

        else:
            return ty, val


array_view_arg_handler = ArrayViewArgHandler()
