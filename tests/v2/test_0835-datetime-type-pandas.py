# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pandas = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_from_pandas():
    values = {"time": ["20190902093000", "20190913093000", "20190921200000"]}
    df = pandas.DataFrame(values, columns=["time"])
    df["time"] = pandas.to_datetime(df["time"], format="%Y%m%d%H%M%S")
    array = ak.layout.NumpyArray(df)
    assert ak.to_list(array) == [
        np.datetime64("2019-09-02T09:30:00"),
        np.datetime64("2019-09-13T09:30:00"),
        np.datetime64("2019-09-21T20:00:00"),
    ]
    array2 = ak.Array(df.values)
    assert ak.to_list(array2) == [
        np.datetime64("2019-09-02T09:30:00"),
        np.datetime64("2019-09-13T09:30:00"),
        np.datetime64("2019-09-21T20:00:00"),
    ]
