# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare
cpp17 = hasattr(ROOT.std, "optional")


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_EmptyArray(flatlist_as_rvec):
    array = ak._v2.contents.EmptyArray()

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RegularArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak._v2.contents.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak._v2.contents.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RecordArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )
    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skip(
    reason="AttributeError: 'Record' object has no attribute 'form', 'Record' object has no attribute 'identifier'"
)
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_Record(flatlist_as_rvec):
    array = ak._v2.contents.RecordArray(
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )
    assert isinstance(array[2], ak._v2.record.Record)

    layout = array[2]
    generator = ak._v2._connect.cling.togenerator(
        array.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert layout.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RecordArray_tuple(flatlist_as_rvec):
    array = ak._v2.Array([(1, 2)])

    layout = array.layout
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert layout.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_IndexedArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_IndexedOptionArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ByteMaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_BitMaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_UnmaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak._v2.contents.UnmaskedArray(
            ak._v2.contents.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_UnionArray_NumpyArray(flatlist_as_rvec):
    array = ak._v2.contents.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.NumpyArray(np.array([1, 2, 3], np.int64)),
            ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skip(reason="the test needs an external data file: see the comments")
def test_data_frame_from_json():
    import os
    import json
    from pathlib import Path

    DIR = os.path.dirname(os.path.abspath(__file__))
    path = Path(DIR).parents[0]

    # The JSON data file for this test can be accessed from
    # European Centre for Disease Prevention and Control
    # An agency of the European Union
    # https://www.ecdc.europa.eu/en/publications-data/data-virus-variants-covid-19-eueea
    with open(os.path.join(path, "samples/covid.json")) as f:
        data = json.load(f)

    array = ak._v2.operations.from_iter(data)

    data_frame = ak._v2.to_rdataframe({"variants": array})
    out = ak._v2.from_rdataframe(
        data_frame,
        columns=("variants",),
    )
    assert array.to_list() == out["variants"].to_list()
