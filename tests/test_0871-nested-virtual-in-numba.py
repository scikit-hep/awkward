# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os.path

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


numba = pytest.importorskip("numba")


def test_constructed():
    @numba.njit
    def f1(array):
        out = np.ones(6, np.float64)
        i = 0
        for x in array:
            out[i] = x
            i += 1
        return out

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    virtual1 = ak.layout.VirtualArray(
        ak.layout.ArrayGenerator(
            lambda: content, form=content.form, length=len(content)
        )
    )
    virtual2 = ak.layout.VirtualArray(
        ak.layout.ArrayGenerator(
            lambda: virtual1, form=virtual1.form, length=len(virtual1)
        )
    )
    array = ak.Array(virtual2)

    assert f1(array).tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]


def test_enumerate_Partitioned():
    @numba.njit
    def f1(array):
        for _, _ in enumerate(array):
            pass

    array = ak.repartition(ak.Array(range(100)), 10)
    f1(array)


pytest.importorskip("pyarrow.parquet")


def test_parquet1(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array([{"x": {"y": 0.0}}, {"x": {"y": 1.1}}, {"x": {"y": 2.2}}])
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            out[i] = obj.x.y
            i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]


def test_parquet2(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array([{"x": [{"y": 0.0}]}, {"x": [{"y": 1.1}]}, {"x": [{"y": 2.2}]}])
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            for subobj in obj.x:
                out[i] = subobj.y
                i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]


def test_parquet3(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [[{"x": [{"y": 0.0}]}], [{"x": [{"y": 1.1}]}], [{"x": [{"y": 2.2}]}]]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for sublist in lazy:
            for obj in sublist:
                for subobj in obj.x:
                    out[i] = subobj.y
                    i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]


def test_parquet4(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [
            {"z": [{"x": [{"y": 0.0}]}]},
            {"z": [{"x": [{"y": 1.1}]}]},
            {"z": [{"x": [{"y": 2.2}]}]},
        ]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for sublist in lazy:
            for obj in sublist.z:
                for subobj in obj.x:
                    out[i] = subobj.y
                    i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]


def test_parquet1b(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [
            {"x": {"y": 0.0, "z": 0}},
            {"x": {"y": 1.1, "z": 1}},
            {"x": {"y": 2.2, "z": 2}},
        ]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            out[i] = obj.x.y
            i += 1
        return out

    @numba.njit
    def f2(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            out[i] = obj.x.z
            i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]
    assert f2(lazy).tolist() == [0, 1, 2]


def test_parquet2b(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [
            {"x": [{"y": 0.0, "z": 0}]},
            {"x": [{"y": 1.1, "z": 1}]},
            {"x": [{"y": 2.2, "z": 2}]},
        ]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            for subobj in obj.x:
                out[i] = subobj.y
                i += 1
        return out

    @numba.njit
    def f2(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for obj in lazy:
            for subobj in obj.x:
                out[i] = subobj.z
                i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]
    assert f2(lazy).tolist() == [0, 1, 2]


def test_parquet3b(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [
            [{"x": [{"y": 0.0, "z": 0}]}],
            [{"x": [{"y": 1.1, "z": 1}]}],
            [{"x": [{"y": 2.2, "z": 2}]}],
        ]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for sublist in lazy:
            for obj in sublist:
                for subobj in obj.x:
                    out[i] = subobj.y
                    i += 1
        return out

    @numba.njit
    def f2(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for sublist in lazy:
            for obj in sublist:
                for subobj in obj.x:
                    out[i] = subobj.z
                    i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]
    assert f2(lazy).tolist() == [0, 1, 2]


def test_parquet4b(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    array = ak.Array(
        [
            {"z": [{"x": 0.0, "w": 0}]},
            {"z": [{"x": 1.1, "w": 1}]},
            {"z": [{"x": 2.2, "w": 2}]},
        ]
    )
    ak.to_parquet(array, filename)

    lazy = ak.from_parquet(filename, lazy=True, lazy_cache=None)

    @numba.njit
    def f1(lazy):
        out = np.ones(3, np.float64)
        i = 0
        for sublist in lazy:
            for obj in sublist.z:
                out[i] = obj.x
                i += 1
        return out

    assert f1(lazy).tolist() == [0.0, 1.1, 2.2]
