# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


@pytest.mark.parametrize("extensionarray", [True, False])
def test_size_zero_regular_keeps_length(extensionarray):
    array = ak.Array(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([], dtype=np.float64)), 0, zeros_length=3
        )
    )
    arrow = ak.to_arrow(array, extensionarray=extensionarray)

    result = ak.from_arrow(arrow)
    assert str(result.type) == "3 * 0 * float64"
    assert result.to_list() == [[], [], []]

    assert ak.from_arrow(arrow.slice(1)).to_list() == [[], []]
