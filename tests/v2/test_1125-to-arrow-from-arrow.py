# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")


def test_numpyarray_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    assert ak.to_list(akarray) == ak.to_list(akarray3)
    assert akarray.form.type == akarray3.form.type


def test_numpyarray():
    akarray = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    paarray = akarray.to_arrow()
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_numpyarray_parameters_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3]), parameters={"which": "only"}
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "only"

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    assert ak.to_list(akarray) == ak.to_list(akarray3)
    assert akarray.form.type == akarray3.form.type
    assert akarray3.parameter("which") == "only"


def test_unmaskedarray_numpyarray_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    assert ak.to_list(akarray) == ak.to_list(akarray3)
    assert akarray.form.type == akarray3.form.type


def test_unmaskedarray_numpyarray_parameters_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.UnmaskedArray(
        ak._v2.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3]), parameters={"which": "inner"}
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "outer"
    assert akarray2.content.parameter("which") == "inner"

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    assert ak.to_list(akarray) == ak.to_list(akarray3)
    assert akarray.form.type == akarray3.form.type
    assert akarray3.parameter("which") == "outer"
    assert akarray3.content.parameter("which") == "inner"


def test_numpyarray_bool():
    akarray = ak._v2.contents.NumpyArray(
        np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.bool_)
    )
    paarray = akarray.to_arrow()
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_indexedoptionarray_emptyarray_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.full(14, -1, dtype=np.uint64)),
        ak._v2.contents.EmptyArray(parameters={"which": "inner"}),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "outer"
    assert akarray2.content.parameter("which") == "inner"

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    # table = pyarrow_parquet.read_table(filename)
    # akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    # assert ak.to_list(akarray) == ak.to_list(akarray3)
    # assert akarray.form.type == akarray3.form.type
    # assert akarray3.parameter("which") == "outer"
    # assert akarray3.content.parameter("which") == "inner"


def test_indexedoptionarray_numpyarray_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(
            np.array([-1, 5, -1, 4, 3, 2, 1, -1, 0, -1], dtype=np.uint64)
        ),
        ak._v2.contents.NumpyArray(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6], parameters={"which": "inner"}
        ),
        parameters={"which": "outer"},
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
    assert akarray2.parameter("which") == "outer"
    assert akarray2.content.parameter("which") == "inner"

    pyarrow_parquet.write_table(pyarrow.table({"": paarray}), filename)
    table = pyarrow_parquet.read_table(filename)
    akarray3 = ak._v2._connect.pyarrow.handle_arrow(table[0].chunks[0])
    assert ak.to_list(akarray) == ak.to_list(akarray3)
    assert akarray.form.type == akarray3.form.type
    assert akarray3.parameter("which") == "outer"
    assert akarray3.content.parameter("which") == "inner"


def test_dictionary_encoding():
    akarray = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64(np.array([3, 2, 2, 2, 0, 1, 3], dtype=np.uint64)),
        ak._v2.contents.NumpyArray([0.0, 1.1, 2.2, 3.3], parameters={"which": "inner"}),
        parameters={"__array__": "categorical", "which": "outer"},
    )
    paarray = akarray.to_arrow(use_extensionarray=False)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray2.parameter("__array__") == "categorical"
    assert akarray2.parameter("which") is None
    assert akarray2.content.parameter("which") is None


@pytest.mark.parametrize("list_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_listoffsetarray(list_to32, dtype):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
    )
    paarray = akarray.to_arrow(list_to32=list_to32)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


@pytest.mark.parametrize("string_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_listoffsetraray_string(string_to32, dtype):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string"},
    )
    paarray = akarray.to_arrow(string_to32=string_to32)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


@pytest.mark.parametrize("bytestring_to32", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_listoffsetraray_bytestring(bytestring_to32, dtype):
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=dtype)),
        ak._v2.contents.NumpyArray(
            np.arange(97, 107, dtype=np.uint8), parameters={"__array__": "byte"}
        ),
        parameters={"__array__": "bytestring"},
    )
    paarray = akarray.to_arrow(bytestring_to32=bytestring_to32)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_listoffsetarray_extensionarray():
    akarray = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index32(np.array([0, 3, 3, 5, 6, 10], dtype=np.int32)),
        ak._v2.contents.NumpyArray([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
    )
    paarray = akarray.to_arrow(use_extensionarray=True)
    assert ak.to_list(akarray) == paarray.to_pylist()
    # akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    # assert ak.to_list(akarray) == ak.to_list(akarray2)
    # assert akarray.form.type == akarray2.form.type
