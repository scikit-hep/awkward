# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_identity():
    a = np.arange(10)
    b = ak.layout.NumpyArray(a)
    b.setidentities()
    assert np.array(b.identities).tolist() == np.arange(10).reshape(-1, 1).tolist()

    assert np.array(b[3]) == a[3]
    assert (
        np.array(b[3:7].identities).tolist()
        == np.arange(10).reshape(-1, 1)[3:7].tolist()
    )
    assert (
        np.array(b[[7, 3, 3, -4]].identities).tolist()
        == np.arange(10).reshape(-1, 1)[[7, 3, 3, -4]].tolist()
    )
    assert (
        np.array(
            b[
                [True, True, True, False, False, False, True, False, True, False]
            ].identities
        ).tolist()
        == np.arange(10)
        .reshape(-1, 1)[
            [True, True, True, False, False, False, True, False, True, False]
        ]
        .tolist()
    )

    assert np.array(b[1:][3]) == a[1:][3]
    assert (
        np.array(b[1:][3:7].identities).tolist()
        == np.arange(10).reshape(-1, 1)[1:][3:7].tolist()
    )
    assert (
        np.array(b[1:][[7, 3, 3, -4]].identities).tolist()
        == np.arange(10).reshape(-1, 1)[1:][[7, 3, 3, -4]].tolist()
    )
    assert (
        np.array(
            b[1:][
                [True, True, False, False, False, True, False, True, False]
            ].identities
        ).tolist()
        == np.arange(10)
        .reshape(-1, 1)[1:][[True, True, False, False, False, True, False, True, False]]
        .tolist()
    )
