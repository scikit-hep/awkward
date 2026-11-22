# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.clongdouble])
@pytest.mark.parametrize("axis", [None, -1, 0])
@pytest.mark.parametrize(
    "name",
    ["sum", "prod", "min", "max", "argmin", "argmax", "any", "all", "count_nonzero"],
)
def test_extended_ragged_reductions(dtype, axis, name):
    # Exact small values make the narrower result a meaningful comparison.
    source = [[3, None, 1, 1], [], [None, None], [-0.0, 0.0, -2], [4]]
    if np.dtype(dtype).kind == "c":
        source = [
            [None if x is None else complex(x, 2 if x else 0) for x in row]
            for row in source
        ]
    narrow = ak.Array(source)
    wide = ak.values_astype(narrow, dtype)
    operation = getattr(ak, name)
    ref = operation(narrow, axis=axis, mask_identity=True)
    actual = operation(wide, axis=axis, mask_identity=True)
    assert ak.to_list(actual) == ak.to_list(ref)
    if axis is not None:
        traced = operation(wide.layout.to_typetracer(), axis=axis, mask_identity=True)
        assert actual.layout.form == traced.layout.form


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.clongdouble])
@pytest.mark.parametrize("name", ["sum", "min", "max", "any", "all", "count_nonzero"])
def test_extended_nonfinite_reductions(dtype, name):
    rows = [
        [1.0, np.nan, -2.0],
        [-np.inf, 1.0, np.inf],
        [np.inf, 0.0],
        [np.nan],
        [],
        [None],
    ]
    if np.dtype(dtype).kind == "c":
        rows[0] = [1 + 2j, complex(3, np.nan), -2 + 1j]
        rows[2] = [complex(np.inf, 2), 0j]
    narrow = ak.Array(rows)
    wide = ak.values_astype(narrow, dtype)
    operation = getattr(ak, name)
    with np.errstate(invalid="ignore"):
        actual = ak.to_numpy(operation(wide, axis=-1, mask_identity=True))
        expected = ak.to_numpy(operation(narrow, axis=-1, mask_identity=True))
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
    )
    actual, expected = np.ma.compressed(actual), np.ma.compressed(expected)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)
