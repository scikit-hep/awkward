from demo_impl cimport ArrayBuffers, create_demo_array as create_demo_array_impl
from cython.operator cimport dereference as deref

# Import both types and functions
cimport numpy as np
import numpy as np
import awkward as ak


def create_demo_array():
    array_buffers = create_demo_array_impl()

    buffers = {}
    # Convert from raw buffers to NumPy arrays
    for it in array_buffers.buffers:
        # Pull out name, buffer, and lookup length of buffer
        name = it.first
        buffer = it.second
        nbytes = array_buffers.buffer_nbytes[name]

        # Store buffer in Python dict
        buffers[name.decode('UTF-8')] = np.asarray(<np.uint8_t[:nbytes]>buffer)

    return ak.from_buffers(array_buffers.form.decode('UTF-8'), array_buffers.length, buffers)
