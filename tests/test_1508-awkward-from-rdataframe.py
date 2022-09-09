# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
import sys

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_refcount():
    array = ak._v2.Array([[[1], [2]], [[3], [4, 5]]])

    assert [sys.getrefcount(x) == 2 for x in (array)]

    data_frame = ak._v2.to_rdataframe({"x": array})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    column_type = data_frame.GetColumnType("x")
    result_ptrs = data_frame.Take[column_type]("x")
    view = result_ptrs.begin()
    lookup = result_ptrs.begin().lookup()
    generator = lookup["x"].generator

    column_type = data_frame.GetColumnType("x")
    result_ptrs = data_frame.Take[column_type]("x")
    view = result_ptrs.begin()
    lookup = result_ptrs.begin().lookup()
    generator = lookup["x"].generator

    array_out = ak._v2.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert array.to_list() == array_out["x"].to_list()

    assert [
        sys.getrefcount(x) == 2
        for x in (
            array,
            array_out,
            lookup,
            generator,
            view,
            result_ptrs.begin().lookup(),
        )
    ]

    for _ in range(3):

        def f1(x):
            return 3.14

        for _ in range(10):
            f1(result_ptrs.begin().lookup())
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array,
                    array_out,
                    lookup,
                    generator,
                    view,
                    result_ptrs.begin().lookup(),
                )
            ]

    for _ in range(3):

        def f2(x):
            return x

        for _ in range(10):
            y = f2(result_ptrs.begin().lookup())
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array,
                    array_out,
                    lookup,
                    generator,
                    view,
                    result_ptrs.begin().lookup(),
                )
            ]

    for _ in range(3):

        def f3(x):
            return x, x

        for _ in range(10):
            y = f3(  # noqa: F841 (checking reference count)
                result_ptrs.begin().lookup()
            )
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array,
                    array_out,
                    lookup,
                    generator,
                    view,
                    result_ptrs.begin().lookup(),
                )
            ]


def test_data_frame_vec_of_vec_of_integers():
    ak_array_in = ak._v2.Array([[[1], [2]], [[3], [4, 5]]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


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
