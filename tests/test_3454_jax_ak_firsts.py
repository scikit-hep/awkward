# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

ak.jax.register_and_check()


def test_jax_ak_firsts():
    jax_array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]], backend="jax")
    jax_firsts = ak.firsts(jax_array)
    cpu_array = ak.to_backend(jax_array, "cpu")
    cpu_firsts = ak.firsts(cpu_array)
    assert jax_firsts.to_list() == cpu_firsts.to_list()
