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
    # FIXME:
    array_out = generator.tolayout(lookup, 0, ("y"))
    #   assert array["y"].to_list() == array_out.to_list()
    # E       assert [0.0, 1.1, 2.2, 3.3, 4.4] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    assert array["y"].to_list() == array_out.to_list()

    data_frame_one = ak._v2.to_rdataframe({"one": array})
    assert str(data_frame_one.GetColumnType("one")).startswith(
        "awkward::Record_Something_"
    )
