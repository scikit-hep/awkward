# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", ["U2", "S2"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2),
        (1, 1, 2),
        (2, 3),
        (2, 3, 2),
        (2, 3, 0),
        (0, 3, 0),
        (2, 0, 3),
        (2, 0, 3, 0),
    ],
)
@pytest.mark.parametrize("regulararray", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_multidimensional_masked_strings_without_mask(
    dtype, shape, regulararray, masked
):
    values = np.resize(np.array(["a", "bc"], dtype=dtype), shape)
    array = np.ma.MaskedArray(values) if masked else values

    result = ak.from_numpy(array, regulararray=regulararray)

    assert result.to_list() == values.tolist()
    assert ak.validity_error(result) == ""
    roundtrip = ak.to_numpy(result)
    assert roundtrip.shape == values.shape
    np.testing.assert_array_equal(roundtrip, values)
    np.testing.assert_array_equal(np.ma.getmaskarray(roundtrip), False)
