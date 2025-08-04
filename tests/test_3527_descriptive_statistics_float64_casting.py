# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_int32_overflow():
    np.random.seed(42)
    x = np.random.randint(2**21, 2**22, size=1000, dtype=np.int32)
    y = np.random.randint(2**21, 2**22, size=1000, dtype=np.int32)

    np.testing.assert_allclose(np.sum(x), ak.sum(x))
    np.testing.assert_allclose(np.mean(x), ak.mean(x))
    np.testing.assert_allclose(np.var(x), ak.var(x))
    np.testing.assert_allclose(np.std(x), ak.std(x))
    np.testing.assert_allclose(np.cov(x, y, ddof=0)[0][1], ak.covar(x, y))
    np.testing.assert_allclose(np.corrcoef(x, y)[0][1], ak.corr(x, y))


def test_int64_overflow():
    np.random.seed(42)
    x = np.random.randint(2**61, 2**62, size=1000, dtype=np.int64)
    y = np.random.randint(2**61, 2**62, size=1000, dtype=np.int64)

    np.testing.assert_allclose(np.sum(x), ak.sum(x))
    np.testing.assert_allclose(np.mean(x), ak.mean(x))
    np.testing.assert_allclose(np.var(x), ak.var(x))
    np.testing.assert_allclose(np.std(x), ak.std(x))
    np.testing.assert_allclose(np.cov(x, y, ddof=0)[0][1], ak.covar(x, y))
    np.testing.assert_allclose(np.corrcoef(x, y)[0][1], ak.corr(x, y))
