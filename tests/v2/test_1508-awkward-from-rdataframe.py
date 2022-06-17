# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_NumpyArray():
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())

    # data_frame = ak._v2.to_rdataframe({"x": array})
    #
    # assert data_frame.GetColumnType("x") == "double"
    #
    # ak_array_out = ak._v2.from_rdataframe(
    #     data_frame, column="x", column_as_record=False
    # )
    assert array.to_list() == array_out.to_list()


def test_data_frame_ListOffsetArray_NumpyArray():
    array = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


def test_nested_ListOffsetArray_NumpyArray():
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.listoffsetarray.ListOffsetArray(
            ak._v2.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()
