# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import cupy as cp
import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("policy", ["same_kind", "equiv", "family"])
@pytest.mark.parametrize("mergebool", [False, True])
def test_cuda_form_and_layout_mergeability(policy, mergebool):
    dtypes = (
        "bool",
        "int8",
        "int32",
        "int64",
        "uint32",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    for left, right in itertools.product(dtypes, repeat=2):
        a = ak.contents.NumpyArray(cp.empty(0, dtype=left))
        b = ak.contents.NumpyArray(cp.empty(0, dtype=right))
        expected = ak._do.mergeable(a.form, b.form, mergebool, policy)
        assert ak._do.mergeable(a, b, mergebool, policy) is expected
        assert ak._do.mergeable(a.form, b, mergebool, policy) is expected
        assert ak._do.mergeable(a, b.form, mergebool, policy) is expected
        if policy != "family":
            assert bool(cp.can_cast(left, right, casting=policy)) is bool(
                np.can_cast(left, right, casting=policy)
            )
