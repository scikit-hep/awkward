# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

# The float64 accumulator used by ak.mean/var/std/corr/covar must also work for
# boolean input on the typetracer backend (as it does on concrete arrays).


@pytest.mark.parametrize("forget_length", [False, True])
@pytest.mark.parametrize(
    "func",
    [
        lambda x: ak.mean(x, axis=-1),
        lambda x: ak.mean(x, axis=None),
        lambda x: ak.var(x, axis=-1),
        lambda x: ak.std(x, axis=0),
        lambda x: ak.corr(x, x, axis=-1),
        lambda x: ak.covar(x, x, axis=None),
    ],
)
def test_bool_statistics_typetracer(func, forget_length):
    array = ak.Array([[True, False, True], [], [False]])
    typetracer = ak.Array(array.layout.to_typetracer(forget_length=forget_length))

    expected = func(array)
    result = func(typetracer)

    if isinstance(expected, ak.Array):
        assert result.layout.form == expected.layout.form
    else:
        assert result.dtype == np.dtype(np.float64)
