# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare
cpp17 = hasattr(ROOT.std, "optional")


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_EmptyArray(flatlist_as_rvec):
    array = ak.contents.EmptyArray()

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_NumpyArray(flatlist_as_rvec):
    array = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RegularArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])),
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak.contents.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RecordArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )
    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RecordArray_tuple(flatlist_as_rvec):
    array = ak.Array([(1, 2)])

    layout = array.layout
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert layout.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_IndexedArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_IndexedOptionArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ByteMaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_BitMaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.BitMaskedArray(
        ak.index.Index(
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
        ak.contents.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_UnmaskedArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.UnmaskedArray(
            ak.contents.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_UnionArray_NumpyArray(flatlist_as_rvec):
    array = ak.contents.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.from_iter(["1", "2", "3"], highlevel=False),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    layout = array
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


@pytest.mark.skip(reason="the test needs an external data file: see the comments")
def test_data_frame_from_json():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "samples" / "covid.json"

    # The JSON data file for this test can be accessed from
    # European Centre for Disease Prevention and Control
    # An agency of the European Union
    # https://www.ecdc.europa.eu/en/publications-data/data-virus-variants-covid-19-eueea
    with path.open() as f:
        data = json.load(f)

    array = ak.operations.from_iter(data)

    data_frame = ak.to_rdataframe({"variants": array})
    out = ak.from_rdataframe(
        data_frame,
        columns=("variants",),
    )
    assert array.to_list() == out["variants"].to_list()
