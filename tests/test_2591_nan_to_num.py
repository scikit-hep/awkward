# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pytest.importorskip("jax")

ak.jax.register_and_check()


def test():
    ak.nan_to_num(
        ak.Array([1.1, 2.2, 3.3], backend="jax"),
        nan=ak.Array([1.1, 2.2, 3.3], backend="jax"),
    )
