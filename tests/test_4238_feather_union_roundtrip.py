# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import os

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.feather")

# pyarrow < 26 drops the buffers of unions wrapped in extension types when
# writing the Arrow IPC format (apache/arrow#50623)
pyarrow_writes_unions = parse_version(pyarrow.__version__) >= parse_version("26.0.0")


def union_array():
    return ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 1], dtype=np.int8)),
            ak.index.Index64(np.array([0, 0], dtype=np.int64)),
            [
                ak.contents.NumpyArray(np.array([1.5], dtype=np.float64)),
                ak.to_layout(["abc"]),
            ],
        )
    )


@pytest.mark.skipif(not pyarrow_writes_unions, reason="requires pyarrow >= 26")
def test_union(tmp_path):
    filename = os.path.join(tmp_path, "union.feather")
    array = union_array()

    ak.to_feather(array, filename)
    assert ak.array_equal(ak.from_feather(filename), array)


@pytest.mark.skipif(pyarrow_writes_unions, reason="requires pyarrow < 26")
def test_union_warns_on_old_pyarrow(tmp_path):
    filename = os.path.join(tmp_path, "union.feather")
    array = union_array()

    with pytest.warns(UserWarning, match="apache/arrow#50623"):
        ak.to_feather(array, filename)
    with pytest.warns(UserWarning, match="apache/arrow#50623"):
        ak.to_feather(ak.Array([{"x": array}]), filename)

    # without extension types, the file is valid
    ak.to_feather(array, filename, extensionarray=False)
    assert ak.from_feather(filename).to_list() == [1.5, "abc"]


@pytest.mark.parametrize("extensionarray", [True, False])
def test_empty_union_in_list(tmp_path, extensionarray):
    # the IPC reader returns the buffers of the zero-length union as None
    filename = os.path.join(tmp_path, "empty-union.feather")
    array = ak.to_packed(ak.Array([[1, "a"], ["b"]])[:, :0])
    assert str(array.type) == "2 * var * union[int64, string]"

    if extensionarray and not pyarrow_writes_unions:
        with pytest.warns(UserWarning, match="apache/arrow#50623"):
            ak.to_feather(array, filename, extensionarray=extensionarray)
        return

    ak.to_feather(array, filename, extensionarray=extensionarray)
    result = ak.from_feather(filename)
    assert result.to_list() == [[], []]
    assert result.type == array.type
