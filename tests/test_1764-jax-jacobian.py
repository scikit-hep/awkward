# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()


@pytest.mark.skip("Jacobian support not implemented")
def test():
    array = ak.Array([[1, 2, 3], [4, 5, 6.0]])

    def func(x):
        return x * 2 - 1

    array_np = ak.to_numpy(array)
    jac_np = jax.jacfwd(func)(array_np)

    jac = jax.jacfwd(func)(array)
    assert jac.to_list() == jac_np.to_list()
