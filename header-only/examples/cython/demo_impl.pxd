from libcpp.map cimport map
from libcpp.string cimport string

# Declare out exported types
cdef extern from "demo_impl.h":
    cdef cppclass ArrayBuffers:
        map[string, void*] buffers
        map[string, size_t] buffer_nbytes
        string form
        size_t length

    ArrayBuffers create_demo_array()
