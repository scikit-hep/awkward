# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_RecordArray_NumpyArray():
    array = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )
    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._v2._lookup.Lookup(layout)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()
    array_out = generator.tolayout(lookup, 0, ("x"))
    assert array["x"].to_list() == array_out.to_list()
    array_out = generator.tolayout(lookup, 0, ("y"))
    # [0.0, 1.1, 2.2, 3.3, 4.4] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]
    assert array["y"].to_list() == array_out[: len(array["y"])].to_list()

    data_frame_one = ak._v2.to_rdataframe({"one": array})
    assert str(data_frame_one.GetColumnType("one")).startswith(
        "awkward::Record_Something_"
    )


def test_IndexedOptionArray_NumpyArray():
    array = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._v2._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    array_out = generator.tolayout(lookup, 0, ())
    assert array.to_list() == array_out.to_list()


def test_UnionArray_NumpyArray():
    array = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    layout = array
    generator = ak._v2._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
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
        column="variants",
    )
    assert array.to_list() == out["variants"].to_list()
