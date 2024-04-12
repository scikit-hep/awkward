# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

numba = pytest.importorskip("numba")


def test():
    behavior = {}

    ak.jax.register_and_check()

    input_arr = ak.Array([1.0], backend="jax")

    @numba.vectorize(
        [
            numba.float32(numba.float32, numba.float32),
            numba.float64(numba.float64, numba.float64),
        ]
    )
    def _some_kernel(x, y):
        return x * x + y * y

    @ak.mixin_class(behavior)
    class SomeClass:
        @property
        def some_kernel(self):
            return _some_kernel(self.x, self.y)

    ak.behavior.update(behavior)

    arr = ak.zip({"x": input_arr, "y": input_arr}, with_name="SomeClass")

    assert ak.all(arr.some_kernel == ak.Array([2.0], backend="jax"))
