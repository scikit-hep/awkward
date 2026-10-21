# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", ["U2", "S2"])
@pytest.mark.parametrize("shape", [(1, 2), (1, 1, 2)])
@pytest.mark.parametrize("regulararray", [False, True])
def test_multidimensional_masked_strings_without_mask(dtype, shape, regulararray):
    values = np.array(["a", "bc"], dtype=dtype).reshape(shape)
    array = np.ma.MaskedArray(values)

    result = ak.from_numpy(array, regulararray=regulararray)

    assert result.to_list() == values.tolist()
    assert ak.validity_error(result) == ""
    roundtrip = ak.to_numpy(result)
    np.testing.assert_array_equal(roundtrip, values)
    np.testing.assert_array_equal(np.ma.getmaskarray(roundtrip), False)
