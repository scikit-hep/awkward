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


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("width", [7, 8, 9, 127, 128, 129, 1025])
@pytest.mark.parametrize("pattern", ["finite", "boundary", "last", "all", "infinity"])
def test_fused_block_boundaries_match_masked_reduction(dtype, width, pattern):
    data = np.full((3, width), 0.1 + 0.1j, dtype=dtype)
    if pattern == "boundary":
        data[:, 7::8] = complex(1, np.nan)
        data[:, 8::8] = complex(np.nan, 1)
    elif pattern == "last":
        data[:, -1] = complex(1, np.nan)
    elif pattern == "all":
        data[:] = complex(np.nan, 1)
    elif pattern == "infinity":
        data[:, 0] = complex(np.inf, 1)
        data[:, -1] = complex(-np.inf, 1)
    backing = np.concatenate(([999j], data.ravel(), [999j])).astype(dtype)
    source = ak.contents.ListOffsetArray(
        ak.index.Index64(np.arange(1, data.size + 2, width)),
        ak.contents.NumpyArray(backing),
    )
    for operation, ordinary in [(ak.nansum, ak.sum), (ak.nanmean, ak.mean)]:
        expected = ordinary(ak.nan_to_none(source), axis=-1, mask_identity=True)
        actual = operation(source, axis=-1, mask_identity=True)
        assert actual.layout.form == expected.layout.form
        a, b = ak.to_numpy(actual), ak.to_numpy(expected)
        np.testing.assert_array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))
        np.testing.assert_array_equal(np.ma.compressed(a), np.ma.compressed(b))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_fused_blocks_preserve_cancellation_order(dtype):
    rng = np.random.default_rng(4549)
    data = (rng.normal(size=(4, 1025)) + 1j * rng.normal(size=(4, 1025))).astype(dtype)
    data[:, 7::16] = 1e10 + 1e10j
    data[:, 8::16] = -1e10 - 1e10j
    data[1, 31::32] = complex(1, np.nan)
    data[2, 32::32] = complex(np.nan, 1)
    data[3, :16] = complex(-0.0, -0.0)
    expected = ak.to_numpy(ak.sum(ak.nan_to_none(data), axis=-1))
    actual = ak.to_numpy(ak.nansum(data, axis=-1))
    integer = np.uint32 if dtype is np.complex64 else np.uint64
    np.testing.assert_array_equal(actual.view(integer), expected.view(integer))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("width", [8, 31, 127, 128, 129])
def test_fused_preserves_zero_and_subnormal_bits(dtype, width):
    component = np.float32 if dtype is np.complex64 else np.float64
    smallest = np.finfo(component).smallest_subnormal
    data = np.zeros((4, width), dtype=dtype)
    data.real[:, ::2] = -smallest
    data.imag[:, ::2] = smallest
    data.real[0, 1::2] = -0.0
    data.imag[1, 1::2] = -0.0
    data[2, 1::3] = complex(1, np.nan)
    data[3, 2::3] = complex(np.nan, 1)
    expected = ak.to_numpy(ak.sum(ak.nan_to_none(data), axis=-1))
    actual = ak.to_numpy(ak.nansum(data, axis=-1))
    integer = np.uint32 if dtype is np.complex64 else np.uint64
    np.testing.assert_array_equal(actual.view(integer), expected.view(integer))
