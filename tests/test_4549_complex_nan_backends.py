# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("jax")
ak.jax.register_and_check()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("has_nan", [False, True])
def test_jax_complex_nan_mask(dtype, has_nan):
    data = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=dtype)
    if has_nan:
        data[1:] = [complex(3, np.nan), complex(np.nan, 6)]
    source = ak.to_backend(ak.Array(data), "jax")
    result = ak.nan_to_none(source)
    traced = ak.nan_to_none(source.layout.to_typetracer(forget_length=True))
    assert result.layout.form == traced.layout.form
    assert ak.to_backend(result, "cpu").to_list() == (
        [1 + 2j, None, None] if has_nan else data.tolist()
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean, ak.nanprod])
@pytest.mark.parametrize("axis", [0, -1])
def test_jax_complex_nan_reductions(dtype, operation, axis):
    data = np.array([[1 + 2j, complex(3, np.nan)], [2 - 1j, 4 + 5j]], dtype=dtype)
    cpu = ak.Array(data)
    source = ak.to_backend(cpu, "jax")
    result = operation(source, axis=axis)
    expected = operation(cpu, axis=axis)
    traced = operation(source.layout.to_typetracer(forget_length=True), axis=axis)
    np.testing.assert_allclose(
        ak.to_numpy(ak.to_backend(result, "cpu")), ak.to_numpy(expected), rtol=1e-6
    )
    assert result.layout.form == expected.layout.form == traced.layout.form
