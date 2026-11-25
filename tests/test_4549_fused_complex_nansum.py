# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from itertools import pairwise

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("offsets", [[0, 0, 4, 7], [1, 3, 3, 6], [0]])
def test_fused_complex_kernel_matches_reference(dtype, offsets):
    values = np.array(
        [
            1 + 2j,
            complex(1, np.nan),
            complex(np.nan, 2),
            3 - 1j,
            complex(np.inf, 2),
            complex(-np.inf, 3),
            0j,
        ],
        dtype=dtype,
    )
    outlength = len(offsets) - 1
    output = np.full(outlength + 2, 99 + 99j, dtype=dtype)
    counts = np.full(outlength + 2, -99, dtype=np.int64)
    offset_array = np.array(offsets, dtype=np.int64)
    component = np.float32 if dtype is np.complex64 else np.float64
    backend = ak.to_layout(values).backend
    backend.maybe_kernel_error(
        backend[
            "awkward_reduce_nansum_count_complex",
            component,
            np.int64,
            component,
            np.int64,
        ](
            output.view(component),
            counts,
            values.view(component),
            offset_array,
            outlength,
        )
    )
    expected = np.zeros(outlength, dtype=dtype)
    expected_counts = np.zeros(outlength, dtype=np.int64)
    # Add components separately, in canonical segment order.
    with np.errstate(invalid="ignore"):
        for row, (start, stop) in enumerate(pairwise(offsets)):
            real = component(0)
            imag = component(0)
            for value in values[start:stop]:
                if not np.isnan(value):
                    real += value.real
                    imag += value.imag
                    expected_counts[row] += 1
            expected.real[row] = real
            expected.imag[row] = imag
    np.testing.assert_array_equal(output[:outlength], expected)
    np.testing.assert_array_equal(counts[:outlength], expected_counts)
    np.testing.assert_array_equal(output[outlength:], [99 + 99j, 99 + 99j])
    np.testing.assert_array_equal(counts[outlength:], [-99, -99])
