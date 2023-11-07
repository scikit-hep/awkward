# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test_mixed_known_lengths():
    first = ak.from_numpy(np.arange(3 * 4).reshape(3, 4), highlevel=False)
    first_tt = first.to_typetracer(forget_length=False)

    second = ak.from_numpy(6 - np.arange(3 * 4).reshape(3, 4) * 2, highlevel=False)
    second_tt = second.to_typetracer(forget_length=True)

    result_tt = ak.concatenate((first_tt, second_tt), axis=1)
    result = ak.concatenate((first, second), axis=1)

    assert result_tt.layout.form == result.layout.form
    assert result_tt.layout.length == result.layout.length == 3


def test_known_lengths():
    first = ak.from_numpy(np.arange(3 * 4).reshape(3, 4), highlevel=False)
    first_tt = first.to_typetracer(forget_length=False)

    second = ak.from_numpy(6 - np.arange(3 * 4).reshape(3, 4) * 2, highlevel=False)
    second_tt = second.to_typetracer(forget_length=False)

    result_tt = ak.concatenate((first_tt, second_tt), axis=1)
    result = ak.concatenate((first, second), axis=1)

    assert result_tt.layout.form == result.layout.form
    assert result_tt.layout.length == result.layout.length == 3


def test_unknown_lengths():
    first = ak.from_numpy(np.arange(3 * 4).reshape(3, 4), highlevel=False)
    first_tt = first.to_typetracer(forget_length=True)

    second = ak.from_numpy(6 - np.arange(3 * 4).reshape(3, 4) * 2, highlevel=False)
    second_tt = second.to_typetracer(forget_length=True)

    result_tt = ak.concatenate((first_tt, second_tt), axis=1)
    result = ak.concatenate((first, second), axis=1)

    assert result_tt.layout.form == result.layout.form
    assert result_tt.layout.length is unknown_length
