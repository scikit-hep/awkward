# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

dak = pytest.importorskip("dask_awkward")
vector = pytest.importorskip("vector")
vector.register_awkward()


def test():
    a = ak.Array([1.0])
    da = dak.from_awkward(a, 1)
    dv1 = dak.with_name(dak.zip({"x": da, "y": da, "z": da}), "Vector3D")

    result1 = (dv1 + dv1).compute()
    assert result1.tolist() == [{"x": 2, "y": 2, "z": 2}]
    assert str(result1.type).startswith("1 * Vector3D[")
    assert type(result1).__name__ == "VectorArray3D"

    dv2 = dak.with_name(dak.zip({"rho": da, "phi": da, "theta": da}), "Vector3D")
    result2 = (dv2 + dv2).compute()
    assert result2.tolist() == [{"rho": 2, "phi": 1, "theta": 1}]
    assert str(result2.type).startswith("1 * Vector3D[")
    assert type(result2).__name__ == "VectorArray3D"
