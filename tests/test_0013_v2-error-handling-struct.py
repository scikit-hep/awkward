# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v1_to_v2_index

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_numpyarray():
    array = ak.layout.NumpyArray(np.arange(10) * 1.1)
    array = v1_to_v2(array)

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 3]

    with pytest.raises(IndexError):
        array[[5, 3, 20, 8]]

    with pytest.raises(IndexError):
        array[[5, 3, -20, 8]]

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 3]

    with pytest.raises(IndexError):
        array[[5, 3, 20, 8]]

    with pytest.raises(IndexError):
        array[[5, 3, -20, 8]]


def test_listarray_numpyarray():
    starts = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.layout.Index64(np.array([3, 3, 5, 6]))
    content = ak.layout.NumpyArray(np.arange(10) * 1.1)

    starts = v1_to_v2_index(starts)
    stops = v1_to_v2_index(stops)
    content = v1_to_v2(content)
    with pytest.raises(ValueError):
        array = ak._v2.contents.listarray.ListArray(starts, stops, content)

    starts = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.layout.Index64(np.array([3, 3, 5, 6, 10]))
    content = ak.layout.NumpyArray(np.arange(10) * 1.1)
    array = ak.layout.ListArray64(starts, stops, content)
    array = v1_to_v2(array)

    with pytest.raises(IndexError):
        array[20]

    with pytest.raises(IndexError):
        array[-20]

    array[-20:20]

    with pytest.raises(IndexError):
        array[
            20,
        ]

    with pytest.raises(IndexError):
        array[
            -20,
        ]

    array[
        -20:20,
    ]

    with pytest.raises(IndexError):
        array[2, 1, 0]

    with pytest.raises(IndexError):
        array[[2, 0, 0, 20, 3]]

    with pytest.raises(IndexError):
        array[[2, 0, 0, -20, 3]]

    starts = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.layout.Index64(np.array([3, 3, 5, 6, 10]))
    content = ak.layout.NumpyArray(np.arange(10) * 1.1)
    array = ak.layout.ListArray64(starts, stops, content)
    array = v1_to_v2(array)

    with pytest.raises(ValueError):
        array[2, 20]

    with pytest.raises(ValueError):
        array[2, -20]

    with pytest.raises(ValueError):
        array[1:][2, 20]

    with pytest.raises(ValueError):
        array[1:][2, -20]

    with pytest.raises(ValueError):
        array[2, [1, 0, 0, 20]]

    with pytest.raises(ValueError):
        array[2, [1, 0, 0, -20]]

    with pytest.raises(ValueError):
        array[1:][2, [0, 20]]

    with pytest.raises(ValueError):
        array[1:][2, [0, -20]]


def test_listarray_listarray_numpyarray():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    stops1 = ak.layout.Index64(np.array([3, 3, 5, 6, 9]))
    starts2 = ak.layout.Index64(np.array([0, 2, 3, 3]))
    stops2 = ak.layout.Index64(np.array([2, 3, 3, 5]))

    array1 = ak.layout.ListArray64(starts1, stops1, content)
    array2 = ak.layout.ListArray64(starts2, stops2, array1)

    array1 = v1_to_v2(array1)
    array2 = v1_to_v2(array2)

    with pytest.raises(IndexError):
        array2[20]

    with pytest.raises(IndexError):
        array2[
            20,
        ]

    with pytest.raises(ValueError):
        array2[2, 20]

    with pytest.raises(IndexError):
        array2[-20]

    with pytest.raises(IndexError):
        array2[
            -20,
        ]

    with pytest.raises(ValueError):
        array2[2, -20]

    with pytest.raises(ValueError):
        array2[1, 0, 20]

    with pytest.raises(IndexError):
        array2[20]

    with pytest.raises(IndexError):
        array2[
            20,
        ]

    with pytest.raises(ValueError):
        array2[2, 20]

    with pytest.raises(ValueError):
        array2[1:][2, 20]

    with pytest.raises(IndexError):
        array2[-20]

    with pytest.raises(IndexError):
        array2[
            -20,
        ]

    with pytest.raises(ValueError):
        array2[2, -20]

    with pytest.raises(ValueError):
        array2[1:][2, -20]

    with pytest.raises(ValueError):
        array2[1, 0, 20]

    with pytest.raises(ValueError):
        array2[1:][2, 0, 20]

    with pytest.raises(ValueError):
        array2[:, 1:][3, 0, 20]
