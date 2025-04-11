# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import pytest

import awkward as ak

jax = pytest.importorskip("jax")


def test_jax_ak_firsts():
    ak.jax.register_and_check()

    jax_array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]], backend="jax")
    jax_firsts = ak.firsts(jax_array)
    cpu_array = ak.to_backend(jax_array, "cpu")
    cpu_firsts = ak.firsts(cpu_array)
    assert jax_firsts.to_list() == cpu_firsts.to_list()
