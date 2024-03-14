# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest as _

import awkward as ak
from awkward.types.numpytype import NumpyType

def test_complex_ops():
    """
    Testing Awkward implementations (cpu backend) of real, imag, and angle
    """
    arr_1 = ak.Array([[1+0.1j, 2+0.2j, 3+0.3j], [], [4+0.4j, 5+0.5j]])
    arr_csingle = ak.from_numpy(np.array([6.+0.j, 0.+7.j], dtype='complex64'))
    arr_real = ak.from_numpy(np.array([11., 12.], dtype='float16'))

    real_1 = np.real(arr_1)
    assert ak.all(real_1 == ak.Array([[1, 2, 3], [], [4, 5]]))

    real_csingle = np.real(arr_csingle)
    assert ak.all(real_csingle == ak.Array([6., 0.]))
    assert real_csingle.type.content == NumpyType('float32')

    real_real = np.real(arr_real)
    assert ak.all(real_real == ak.Array([11., 12.]))
    assert real_real.type.content == NumpyType('float16')

    imag_1 = np.imag(arr_1)
    assert ak.all(imag_1 == ak.Array([[0.1, 0.2, 0.3], [], [0.4, 0.5]]))

    imag_csingle = np.imag(arr_csingle)
    assert ak.all(imag_csingle == ak.Array([0., 7.]))
    assert imag_csingle.type.content == NumpyType('float32')

    imag_real = np.imag(arr_real)
    assert ak.all(imag_real == ak.Array([0., 0.]))
    assert imag_real.type.content == NumpyType('float16')

    angle_1 = np.angle(arr_1)
    a1 = np.arctan(0.1)
    expected = ak.Array([[a1, a1, a1], [], [a1, a1]])  # or [[a1] * 3, ...]
    assert ak.all(np.abs(angle_1 - expected) < np.finfo(np.float64).eps)

    angle_csingle = np.angle(arr_csingle, deg=True)
    assert ak.all(angle_csingle == ak.Array([0., 90.]))
    assert angle_csingle.type.content == NumpyType('float32')

    angle_real = np.angle(arr_real)
    assert ak.all(angle_real == ak.Array([0., 0.]))
    assert angle_real.type.content == NumpyType('float16')
    assert np.angle(ak.Array([1,2])).type.content == NumpyType('float64')
