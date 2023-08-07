# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

pytest.importorskip("jax")

ak.jax.register_and_check()


def test():
    ak.nan_to_num(
        ak.Array([1.1, 2.2, 3.3], backend="jax"),
        nan=ak.Array([1.1, 2.2, 3.3], backend="jax"),
    )
