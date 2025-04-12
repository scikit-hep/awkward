# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
ak.jax.register_and_check()


def test_jax_ak_firsts():
    jax_array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]], backend="jax")
    jax_firsts = ak.firsts(jax_array)
    cpu_array = ak.to_backend(jax_array, "cpu")
    cpu_firsts = ak.firsts(cpu_array)
    assert jax_firsts.to_list() == cpu_firsts.to_list()


def test_jax_ak_unflatten():
    original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]], backend="jax")
    jax_counts = ak.num(original)
    jax_array = ak.flatten(original)
    jax_unflatten = ak.unflatten(jax_array, jax_counts)
    cpu_counts = ak.to_backend(jax_counts, "cpu")
    cpu_array = ak.to_backend(jax_array, "cpu")
    cpu_unflatten = ak.unflatten(cpu_array, cpu_counts)
    assert jax_unflatten.to_list() == cpu_unflatten.to_list()


def test_jax_run_lengths():
    jax_array = ak.Array([1.1, 1.1, 1.1, 2.2, 3.3, 3.3, 4.4, 4.4, 5.5], backend="jax")
    jax_run_lengths = ak.run_lengths(jax_array)
    cpu_array = ak.to_backend(jax_array, "cpu")
    cpu_run_lengths = ak.run_lengths(cpu_array)
    assert jax_run_lengths.to_list() == cpu_run_lengths.to_list()


def test_jax_listarray_to_listoffsetarray64():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    cpu_listarray = ak.contents.ListArray(starts, stops, content)
    jax_listarray = ak.to_backend(cpu_listarray, "jax", highlevel=False)
    cpu_listoffsetarray = ak.Array(cpu_listarray.to_ListOffsetArray64())
    jax_listoffsetarray = ak.Array(jax_listarray.to_ListOffsetArray64())
    assert cpu_listoffsetarray.to_list() == jax_listoffsetarray.to_list()
