# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.types import ListType
from awkward.types.numpytype import NumpyType


def test_complex_ops():
    """
    Testing Awkward implementations (cpu backend) of real, imag, and angle
    """
    arr_1 = ak.Array([[1 + 0.1j, 2 + 0.2j, 3 + 0.3j], [], [4 + 0.4j, 5 + 0.5j]])
    arr_csingle = ak.from_numpy(np.array([6.0 + 0.0j, 0.0 + 7.0j], dtype="complex64"))
    arr_real = ak.Array([11.0, 12.0])

    real_1 = np.real(arr_1)
    assert ak.all(real_1 == ak.Array([[1, 2, 3], [], [4, 5]]))

    real_csingle = np.real(arr_csingle)
    assert ak.all(real_csingle == ak.Array([6.0, 0.0]))
    assert real_csingle.type.content == NumpyType("float32")

    real_real = np.real(arr_real)
    assert ak.all(real_real == ak.Array([11.0, 12.0]))

    imag_1 = np.imag(arr_1)
    assert ak.all(imag_1 == ak.Array([[0.1, 0.2, 0.3], [], [0.4, 0.5]]))

    imag_csingle = np.imag(arr_csingle)
    assert ak.all(imag_csingle == ak.Array([0.0, 7.0]))
    assert imag_csingle.type.content == NumpyType("float32")

    imag_real = np.imag(arr_real)
    assert ak.all(imag_real == ak.Array([0.0, 0.0]))

    angle_1 = np.angle(arr_1)
    a1 = np.arctan(0.1)
    expected = ak.Array([[a1, a1, a1], [], [a1, a1]])  # or [[a1] * 3, ...]
    assert ak.all(np.abs(angle_1 - expected) < np.finfo(np.float64).eps)

    angle_csingle = np.angle(arr_csingle, deg=True)
    assert ak.all(angle_csingle == ak.Array([0.0, 90.0]))
    assert angle_csingle.type.content == NumpyType("float32")

    angle_real = np.angle(arr_real)
    assert ak.all(angle_real == ak.Array([0.0, 0.0]))


def test_complex_typetracer():
    tt_arr = ak.to_backend(
        ak.Array([[1 + 0.1j, 2 + 0.2j, 3 + 0.3j], [], [4 + 0.4j, 5 + 0.5j]]),
        "typetracer",
    )
    tt_csingle = ak.to_backend(
        ak.from_numpy(np.array([6.0 + 0.0j, 0.0 + 7.0j], dtype="complex64")),
        "typetracer",
    )
    tt_real = ak.to_backend(
        ak.from_numpy(np.array([11.0, 12.0], dtype="float16")), "typetracer"
    )
    tt_int = ak.to_backend(ak.Array([1, 2]), "typetracer")

    real_arr = np.real(tt_arr)
    assert real_arr.type.content == ListType(NumpyType("float64"))
    real_tt_csingle = np.real(tt_csingle)
    assert real_tt_csingle.type.content == NumpyType("float32")
    real_tt_real = np.real(tt_real)
    assert real_tt_real.type.content == NumpyType("float16")

    imag_tt_csingle = np.imag(tt_csingle)
    assert imag_tt_csingle.type.content == NumpyType("float32")
    imag_tt_real = np.imag(tt_real)
    assert imag_tt_real.type.content == NumpyType("float16")

    angle_tt_csingle = np.angle(tt_csingle, deg=True)
    assert angle_tt_csingle.type.content == NumpyType("float32")
    angle_tt_real = np.angle(tt_real)
    assert angle_tt_real.type.content == NumpyType("float16")
    assert np.angle(tt_int).type.content == NumpyType("float64")
