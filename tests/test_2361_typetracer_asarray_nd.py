# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

from awkward._nplikes.typetracer import TypeTracer

typetracer = TypeTracer.instance()


def test_nd():
    data = [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [1, 2, 3],
            [4, 5, 8],
        ],
    ]
    result = typetracer.asarray(data, dtype=np.uint8)
    assert result.dtype == np.dtype(np.uint8)
    assert result.shape == (2, 2, 3)

    # Check default size of array
    array = np.array([1, 2, 3])
    default_int_dtype = array.dtype

    result = typetracer.asarray(data)
    assert result.dtype == default_int_dtype
    assert result.shape == (2, 2, 3)


def test_nd_ragged():
    data = [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [1, 2, 3],
            [4, 5],
        ],
    ]
    with pytest.raises(ValueError, match=r"sequence at dimension .* does not match"):
        typetracer.asarray(data)


def test_unknown_scalar():
    unknown_array = typetracer.asarray([0, 1, 2.0], dtype=np.dtype(np.float64))

    data = [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [1, 2, 3],
            [4, 5, unknown_array[0]],
        ],
    ]
    result = typetracer.asarray(data)
    assert result.dtype == np.dtype(np.float64)
    assert result.shape == (2, 2, 3)
