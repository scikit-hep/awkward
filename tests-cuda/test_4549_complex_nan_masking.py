# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("cupy")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("has_nan", [False, True])
def test_complex_nan_mask_form_matches_typetracer(dtype, has_nan):
    data = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=dtype)
    if has_nan:
        data[1] = complex(3, np.nan)
        data[2] = complex(np.nan, 6)
    source = ak.to_backend(ak.Array(data), "cuda")
    result = ak.nan_to_none(source)
    traced = ak.nan_to_none(
        source.layout.to_typetracer(forget_length=True), highlevel=False
    )
    assert result.layout.form == traced.form
    assert ak.to_backend(result, "cpu").to_list() == (
        [1 + 2j, None, None] if has_nan else data.tolist()
    )
    ak._connect.cuda.synchronize_cuda()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean, ak.nanprod])
def test_complex_nan_reduction_matches_cpu(dtype, operation):
    source = ak.Array(
        np.array(
            [[1 + 2j, complex(3, np.nan)], [complex(np.nan, 1), 4 + 5j]], dtype=dtype
        )
    )
    gpu = ak.to_backend(source, "cuda")
    actual = operation(gpu, axis=-1)
    expected = operation(source, axis=-1)
    traced = operation(gpu.layout.to_typetracer(forget_length=True), axis=-1)
    np.testing.assert_allclose(
        ak.to_numpy(ak.to_backend(actual, "cpu")), ak.to_numpy(expected)
    )
    assert actual.layout.form == expected.layout.form == traced.layout.form
    ak._connect.cuda.synchronize_cuda()
