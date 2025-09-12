# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

try:
    import cuda.cccl.parallel.experimental as parallel

    error_message = None

except ModuleNotFoundError:
    parallel = None
    error_message = """to use {0}, you must install cuda.cccl:

    pip install cuda-cccl

"""


# Reducer: return element with smaller value
def argmin_reducer(x, y):
    return x if x.val < y.val else y


# def argmin_op(i, j, values):
#     """Return index of smaller value."""
#     return i if values[i] <= values[j] else j


# def argmin_reducer(i, j, data):
#     # i and j are indices; data is device array
#     return argmin_op(i, j, data)


def min_op(a, b):
    return a if a < b else b


def min_op_complex(a, b):
    if abs(a) < abs(b):
        return a
    elif abs(b) < abs(a):
        return b
    else:
        if a.real != b.real:
            return a if a.real < b.real else b
        else:
            return a if a.imag < b.imag else b
