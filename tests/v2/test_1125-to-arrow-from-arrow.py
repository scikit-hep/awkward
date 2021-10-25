# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

pyarrow6_is_available = False


def test_numpyarray_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    paarray = akarray.to_arrow()
    if pyarrow6_is_available:
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
    paarray = akarray.to_arrow(use_extensionarray=False)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type


def test_numpyarray_parameters_extensionarray(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    akarray = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3]), parameters={"which": "only"}
    )
    paarray = akarray.to_arrow()
    if pyarrow6_is_available:
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
    paarray = akarray.to_arrow()
    if pyarrow6_is_available:
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
    paarray = akarray.to_arrow()
    if pyarrow6_is_available:
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
    paarray = akarray.to_arrow(use_extensionarray=False)
    assert ak.to_list(akarray) == paarray.to_pylist()
    akarray2 = ak._v2._connect.pyarrow.handle_arrow(paarray)
    assert ak.to_list(akarray) == ak.to_list(akarray2)
    assert akarray.form.type == akarray2.form.type
