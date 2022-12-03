# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")

import awkward._connect.cling  # noqa: E402
import awkward._lookup  # noqa: E402

compiler = ROOT.gInterpreter.Declare
cpp17 = hasattr(ROOT.std, "optional")


def debug_compiler(code):
    print(code)  # noqa: T201
    ROOT.gInterpreter.Declare(code)


def test_NumpyArray():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[1];
  out[2] = obj[3];
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_EmptyArray():
    v2a = ak.contents.emptyarray.EmptyArray()

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
size_t roottest_EmptyArray_v2a(ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  return obj.size();
}}
"""
    )
    assert ROOT.roottest_EmptyArray_v2a(len(layout), lookup.arrayptrs) == 0


def test_NumpyArray_shape():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_shape_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0].size();
  out[3] = obj[0][0][0];
  out[4] = obj[0][0][1];
  out[5] = obj[0][1][0];
  out[6] = obj[0][1][1];
  out[7] = obj[1][0][0];
  out[8] = obj[1][1][1];
}}
"""
    )
    out = np.zeros(9, dtype=np.float64)
    ROOT.roottest_NumpyArray_shape_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [2.0, 3.0, 5.0, 0.0, 1.0, 5.0, 6.0, 15.0, 21.0]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RegularArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RegularArray_NumpyArray_v2a_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0][0];
  out[2] = obj[0][1];
  out[3] = obj[1][0];
  out[4] = obj[1][1];
  out[5] = obj[1].size();
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    getattr(ROOT, f"roottest_RegularArray_NumpyArray_v2a_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [2.0, 0.0, 1.1, 3.3, 4.4, 3.0]


def test_RegularArray_NumpyArray_v2b():
    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
        0,
        zeros_length=10,
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RegularArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[1].size();
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_RegularArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0, 0.0, 0.0]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_ListArray_NumpyArray_v2a_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0];
  out[3] = obj[0][1];
  out[4] = obj[0][2];
  out[5] = obj[1].size();
  out[6] = obj[2].size();
  out[7] = obj[2][0];
  out[8] = obj[2][1];
}}
"""
    )
    out = np.zeros(9, dtype=np.float64)
    getattr(ROOT, f"roottest_ListArray_NumpyArray_v2a_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [3.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_ListOffsetArray_NumpyArray_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0];
  out[3] = obj[0][1];
  out[4] = obj[0][2];
  out[5] = obj[1].size();
  out[6] = obj[2].size();
  out[7] = obj[2][0];
  out[8] = obj[2][1];
  out[9] = obj[3].size();
  out[10] = obj[3][0];
}}
"""
    )
    out = np.zeros(11, dtype=np.float64)
    getattr(ROOT, f"roottest_ListOffsetArray_NumpyArray_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [4.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5, 1.0, 7.7]


def test_RecordArray_NumpyArray():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RecordArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  auto rec1 = obj[1];
  auto rec4 = obj[4];
  out[1] = rec1.x();
  out[2] = rec1.y();
  out[3] = rec4.x();
  out[4] = rec4.y();
}}
"""
    )
    out = np.zeros(5, dtype=np.float64)
    ROOT.roottest_RecordArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RecordArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  auto rec1 = obj[1];
  auto rec4 = obj[4];
  out[1] = rec1.slot0();
  out[2] = rec1.slot1();
  out[3] = rec4.slot0();
  out[4] = rec4.slot1();
}}
"""
    )
    out = np.zeros(5, dtype=np.float64)
    ROOT.roottest_RecordArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2c = ak.contents.recordarray.RecordArray([], [], 10)

    layout = v2c
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RecordArray_NumpyArray_v2c(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  obj[5];
}}
"""
    )
    out = np.zeros(1, dtype=np.float64)
    ROOT.roottest_RecordArray_NumpyArray_v2c(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0]

    v2d = ak.contents.recordarray.RecordArray([], None, 10)

    layout = v2d
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RecordArray_NumpyArray_v2d(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  obj[5];
}}
"""
    )
    out = np.zeros(1, dtype=np.float64)
    ROOT.roottest_RecordArray_NumpyArray_v2d(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0]


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
def test_IndexedArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_IndexedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0];
  out[2] = obj[1];
  out[3] = obj[2];
  out[4] = obj[3];
  out[5] = obj[4];
  out[6] = obj[5];
  out[7] = obj[6];
}}
"""
    )
    out = np.zeros(8, dtype=np.float64)
    ROOT.roottest_IndexedArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [7.0, 2.2, 2.2, 0.0, 1.1, 4.4, 5.5, 4.4]


def test_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_IndexedOptionArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
}}
"""
    )
    out = np.zeros(8, dtype=np.float64)
    ROOT.roottest_IndexedOptionArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [7.0, 2.2, 2.2, 999.0, 1.1, 999.0, 5.5, 4.4]


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
def test_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_ByteMaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_ByteMaskedArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_ByteMaskedArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_ByteMaskedArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
def test_BitMaskedArray_NumpyArray():
    v2a = ak.contents.bitmaskedarray.BitMaskedArray(
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
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_BitMaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_BitMaskedArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_BitMaskedArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_BitMaskedArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
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
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )

    layout = v2c
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_BitMaskedArray_NumpyArray_v2c(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_BitMaskedArray_NumpyArray_v2c(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2d = ak.contents.bitmaskedarray.BitMaskedArray(
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
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )

    layout = v2d
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_BitMaskedArray_NumpyArray_v2d(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_BitMaskedArray_NumpyArray_v2d(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]


@pytest.mark.skipif(not cpp17, reason="ROOT was compiled without C++17 support")
def test_UnmaskedArray_NumpyArray():
    v2a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_UnmaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[2] = obj[3].has_value() ? obj[3].value() : 999.0;
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_UnmaskedArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_UnionArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.from_iter(["1", "2", "3"], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_UnionArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = obj[0].index();
  out[2] = obj[1].index();
  out[3] = obj[2].index();
  out[4] = obj[3].index();
  out[5] = obj[4].index();
  out[6] = obj[5].index();
  out[7] = obj[6].index();
  out[8] = std::get<double>(obj[0]);
  out[9] = std::get<double>(obj[1]);
  out[10] = std::atoi(std::get<std::string>(obj[2]).c_str());
  out[11] = std::atoi(std::get<std::string>(obj[3]).c_str());
  out[12] = std::get<double>(obj[4]);
  out[13] = std::atoi(std::get<std::string>(obj[5]).c_str());
  out[14] = std::get<double>(obj[6]);
}}
"""
    )
    out = np.zeros(15, dtype=np.float64)
    ROOT.roottest_UnionArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [7, 1, 1, 0, 0, 1, 0, 1, 5.5, 4.4, 1, 2, 3.3, 3, 5.5]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([999.0, 0.0, 1.1, 2.2, 3.3]),
            parameters={"some": "stuff", "other": [1, 2, "three"]},
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_NumpyArray_v2a_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[1];
  out[2] = obj[3];
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    getattr(ROOT, f"roottest_nested_NumpyArray_v2a_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_nested_NumpyArray_shape():
    data = np.full((3, 3, 5), 999, dtype=np.int64)
    data[1:3] = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)

    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(data),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_NumpyArray_shape_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0].size();
  out[3] = obj[0][0][0];
  out[4] = obj[0][0][1];
  out[5] = obj[0][1][0];
  out[6] = obj[0][1][1];
  out[7] = obj[1][0][0];
  out[8] = obj[1][1][1];
}}
"""
    )
    out = np.zeros(9, dtype=np.float64)
    ROOT.roottest_nested_NumpyArray_shape_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [2.0, 3.0, 5.0, 0.0, 1.0, 5.0, 6.0, 15.0, 21.0]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_RegularArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak.contents.regulararray.RegularArray(
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 999, 999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
            3,
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RegularArray_NumpyArray_v2a_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0][0];
  out[2] = obj[0][1];
  out[3] = obj[1][0];
  out[4] = obj[1][1];
  out[5] = obj[1].size();
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    getattr(ROOT, f"roottest_nested_RegularArray_NumpyArray_v2a_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [2.0, 0.0, 1.1, 3.3, 4.4, 3.0]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.regulararray.RegularArray(
            ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
            0,
            zeros_length=11,
        ),
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RegularArray_NumpyArray_v2b{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[1].size();
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    getattr(ROOT, f"roottest_nested_RegularArray_NumpyArray_v2b{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [10.0, 0.0, 0.0]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_ListArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 4], dtype=np.int64)),
        ak.contents.listarray.ListArray(
            ak.index.Index(np.array([999, 4, 100, 1], np.int64)),
            ak.index.Index(np.array([999, 7, 100, 3, 200], np.int64)),
            ak.contents.numpyarray.NumpyArray(
                np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
            ),
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_ListArray_NumpyArray_v2a_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0];
  out[3] = obj[0][1];
  out[4] = obj[0][2];
  out[5] = obj[1].size();
  out[6] = obj[2].size();
  out[7] = obj[2][0];
  out[8] = obj[2][1];
}}
"""
    )
    out = np.zeros(9, dtype=np.float64)
    getattr(ROOT, f"roottest_nested_ListArray_NumpyArray_v2a_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [3.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5]


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.listoffsetarray.ListOffsetArray(
            ak.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(
        layout.form, flatlist_as_rvec=flatlist_as_rvec
    )
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_ListOffsetArray_NumpyArray_{flatlist_as_rvec}(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].size();
  out[2] = obj[0][0];
  out[3] = obj[0][1];
  out[4] = obj[0][2];
  out[5] = obj[1].size();
  out[6] = obj[2].size();
  out[7] = obj[2][0];
  out[8] = obj[2][1];
  out[9] = obj[3].size();
  out[10] = obj[3][0];
}}
"""
    )
    out = np.zeros(11, dtype=np.float64)
    getattr(ROOT, f"roottest_nested_ListOffsetArray_NumpyArray_{flatlist_as_rvec}")(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [4.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5, 1.0, 7.7]


def test_nested_RecordArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            ["x", "y"],
            parameters={"__record__": "Something"},
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RecordArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  auto rec1 = obj[1];
  auto rec4 = obj[4];
  out[1] = rec1.x();
  out[2] = rec1.y();
  out[3] = rec4.x();
  out[4] = rec4.y();
}}
"""
    )
    out = np.zeros(5, dtype=np.float64)
    ROOT.roottest_nested_RecordArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            None,
        ),
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RecordArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  auto rec1 = obj[1];
  auto rec4 = obj[4];
  out[1] = rec1.slot0();
  out[2] = rec1.slot1();
  out[3] = rec4.slot0();
  out[4] = rec4.slot1();
}}
"""
    )
    out = np.zeros(5, dtype=np.float64)
    ROOT.roottest_nested_RecordArray_NumpyArray_v2b(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2c = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], [], 11),
    )

    layout = v2c
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RecordArray_NumpyArray_v2c(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  obj[5];
}}
"""
    )
    out = np.zeros(1, dtype=np.float64)
    ROOT.roottest_nested_RecordArray_NumpyArray_v2c(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0]

    v2d = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], None, 11),
    )

    layout = v2d
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_RecordArray_NumpyArray_v2d(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  obj[5];
}}
"""
    )
    out = np.zeros(1, dtype=np.float64)
    ROOT.roottest_nested_RecordArray_NumpyArray_v2d(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [10.0]


def test_nested_IndexedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedarray.IndexedArray(
            ak.index.Index(np.array([999, 2, 2, 0, 1, 4, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_IndexedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0];
  out[2] = obj[1];
  out[3] = obj[2];
  out[4] = obj[3];
  out[5] = obj[4];
  out[6] = obj[5];
  out[7] = obj[6];
}}
"""
    )
    out = np.zeros(8, dtype=np.float64)
    ROOT.roottest_nested_IndexedArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [7.0, 2.2, 2.2, 0.0, 1.1, 4.4, 5.5, 4.4]


def test_nested_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedoptionarray.IndexedOptionArray(
            ak.index.Index(np.array([999, 2, 2, -1, 1, -1, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_IndexedOptionArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
}}
"""
    )
    out = np.zeros(8, dtype=np.float64)
    ROOT.roottest_nested_IndexedOptionArray_NumpyArray_v2a(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [7.0, 2.2, 2.2, 999.0, 1.1, 999.0, 5.5, 4.4]


def test_nested_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([123, 1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=True,
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_ByteMaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_nested_ByteMaskedArray_NumpyArray_v2a(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([123, 0, 1, 0, 1, 0], np.int8)),
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=False,
        ),
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_ByteMaskedArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_nested_ByteMaskedArray_NumpyArray_v2b(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]


def test_nested_BitMaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
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
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=True,
            length=14,
            lsb_order=False,
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_BitMaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_nested_BitMaskedArray_NumpyArray_v2a(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=False,
            length=14,
            lsb_order=False,
        ),
    )

    layout = v2b
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_BitMaskedArray_NumpyArray_v2b(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_nested_BitMaskedArray_NumpyArray_v2b(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2c = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=True,
            length=14,
            lsb_order=True,
        ),
    )

    layout = v2c
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_BitMaskedArray_NumpyArray_v2c(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_nested_BitMaskedArray_NumpyArray_v2c(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2d = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            1,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ]
                )
            ),
            valid_when=False,
            length=14,
            lsb_order=True,
        ),
    )

    layout = v2d
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_BitMaskedArray_NumpyArray_v2d(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].has_value() ? obj[0].value() : 999.0;
  out[2] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[3] = obj[2].has_value() ? obj[2].value() : 999.0;
  out[4] = obj[3].has_value() ? obj[3].value() : 999.0;
  out[5] = obj[4].has_value() ? obj[4].value() : 999.0;
  out[6] = obj[5].has_value() ? obj[5].value() : 999.0;
  out[7] = obj[6].has_value() ? obj[6].value() : 999.0;
  out[8] = obj[7].has_value() ? obj[7].value() : 999.0;
  out[9] = obj[8].has_value() ? obj[8].value() : 999.0;
  out[10] = obj[9].has_value() ? obj[9].value() : 999.0;
  out[11] = obj[10].has_value() ? obj[10].value() : 999.0;
  out[12] = obj[11].has_value() ? obj[11].value() : 999.0;
  out[13] = obj[12].has_value() ? obj[12].value() : 999.0;
}}
"""
    )
    out = np.zeros(14, dtype=np.float64)
    ROOT.roottest_nested_BitMaskedArray_NumpyArray_v2d(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]


def test_nested_UnmaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.unmaskedarray.UnmaskedArray(
            ak.contents.numpyarray.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_UnmaskedArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[1].has_value() ? obj[1].value() : 999.0;
  out[2] = obj[3].has_value() ? obj[3].value() : 999.0;
}}
"""
    )
    out = np.zeros(3, dtype=np.float64)
    ROOT.roottest_nested_UnmaskedArray_NumpyArray_v2a(
        out, len(layout), lookup.arrayptrs
    )
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_nested_UnionArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.unionarray.UnionArray(
            ak.index.Index(np.array([123, 1, 1, 0, 0, 1, 0, 1], np.int8)),
            ak.index.Index(np.array([999, 4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
            [
                ak.from_iter(["1", "2", "3"], highlevel=False),
                ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
            ],
        ),
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_nested_UnionArray_NumpyArray_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()}[1];
  out[0] = obj.size();
  out[1] = obj[0].index();
  out[2] = obj[1].index();
  out[3] = obj[2].index();
  out[4] = obj[3].index();
  out[5] = obj[4].index();
  out[6] = obj[5].index();
  out[7] = obj[6].index();
  out[8] = std::get<double>(obj[0]);
  out[9] = std::get<double>(obj[1]);
  out[10] = std::atoi(std::get<std::string>(obj[2]).c_str());
  out[11] = std::atoi(std::get<std::string>(obj[3]).c_str());
  out[12] = std::get<double>(obj[4]);
  out[13] = std::atoi(std::get<std::string>(obj[5]).c_str());
  out[14] = std::get<double>(obj[6]);
}}
"""
    )
    out = np.zeros(15, dtype=np.float64)
    ROOT.roottest_nested_UnionArray_NumpyArray_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [7, 1, 1, 0, 0, 1, 0, 1, 5.5, 4.4, 1, 2, 3.3, 3, 5.5]


def test_ListArray_strings():
    layout = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )

    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_ListArray_strings(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = (obj[0] == "one");
  out[2] = (obj[1] == "two");
  out[3] = (obj[2] == "three");
  out[4] = (obj[3] == "four");
  out[5] = (obj[4] == "five");
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_ListArray_strings(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_RegularArray_strings():
    layout = ak.operations.to_regular(
        ak.operations.from_iter(["onexx", "twoxx", "three", "fourx", "fivex"]),
        axis=1,
        highlevel=False,
    )

    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_RegularArray_strings(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  out[0] = obj.size();
  out[1] = (obj[0] == "onexx");
  out[2] = (obj[1] == "twoxx");
  out[3] = (obj[2] == "three");
  out[4] = (obj[3] == "fourx");
  out[5] = (obj[4] == "fivex");
}}
"""
    )
    out = np.zeros(6, dtype=np.float64)
    ROOT.roottest_RegularArray_strings(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [5.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_NumpyArray_iterator():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_iterator_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};

  size_t i = 0;
  for (auto it = obj.begin();  it != obj.end();  ++it) {{
    out[i] = *it;
    i++;
  }}
}}
"""
    )
    out = np.zeros(4, dtype=np.float64)
    ROOT.roottest_NumpyArray_iterator_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [0.0, 1.1, 2.2, 3.3]


def test_NumpyArray_iterator2():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_iterator2_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};

  size_t i = 0;
  for (auto x : obj) {{
    out[i] = x;
    i++;
  }}
}}
"""
    )
    out = np.zeros(4, dtype=np.float64)
    ROOT.roottest_NumpyArray_iterator2_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [0.0, 1.1, 2.2, 3.3]


def test_NumpyArray_riterator():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout = v2a
    generator = ak._connect.cling.togenerator(layout.form, flatlist_as_rvec=False)
    lookup = ak._lookup.Lookup(layout, generator)
    generator.generate(compiler)

    ROOT.gInterpreter.Declare(
        f"""
void roottest_NumpyArray_riterator_v2a(double* out, ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};

  size_t i = 0;
  for (auto it = obj.rbegin();  it != obj.rend();  ++it) {{
    out[i] = *it;
    i++;
  }}
}}
"""
    )
    out = np.zeros(4, dtype=np.float64)
    ROOT.roottest_NumpyArray_riterator_v2a(out, len(layout), lookup.arrayptrs)
    assert out.tolist() == [3.3, 2.2, 1.1, 0.0]
