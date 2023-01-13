# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_byteorder():
    array = ak.Array([[[1, 2, 3], [4, 5], None, "hi"]])

    _, _, container_little = ak.to_buffers(array, byteorder="<")
    _, _, container_big = ak.to_buffers(array, byteorder=">")

    for name, buffer in container_little.items():
        assert buffer.tobytes() == container_big[name].byteswap().tobytes()


def test_byteorder_default():
    array = ak.Array([[[1, 2, 3], [4, 5], None, "hi"]])

    _, _, container_little = ak.to_buffers(array, byteorder="<")
    _, _, container_default = ak.to_buffers(array)

    for name, buffer in container_little.items():
        assert buffer.tobytes() == container_default[name].tobytes()
