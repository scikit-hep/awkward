from demo_impl cimport ArrayBuffers, create_demo_array as create_demo_array_impl
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray

# Import both types and functions
cimport numpy as np
import numpy as np
import awkward as ak


cdef create_array_view(void* buffer, size_t nbytes):
    cdef cvarray view = <np.uint8_t[:nbytes]> buffer
    # When this view leaves scope, call `free` on the data
    # see https://github.com/cython/cython/blob/05f7a479f6417716b3de2a9559f2724013af6eba/Cython/Utility/MemoryView.pyx#L215
    view.free_data = True
    return np.asarray(view)


def create_demo_array():
    array_buffers = create_demo_array_impl()

    buffers = {}
    # Convert from raw buffers to NumPy arrays
    for it in array_buffers.buffers:
        # Pull out name, buffer, and lookup length of buffer
        name = it.first
        buffer = it.second
        nbytes = array_buffers.buffer_nbytes[name]

        # Create Cython array over buffer
        buffers[name.decode('UTF-8')] = create_array_view(buffer, nbytes)

    return ak.from_buffers(array_buffers.form.decode('UTF-8'), array_buffers.length, buffers)
