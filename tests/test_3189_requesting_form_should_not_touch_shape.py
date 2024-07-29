# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_example():
    layout, report = ak.typetracer.typetracer_with_report(
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "node1",
            },
            "form_key": "node0",
        }
    )
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_NumpyArray():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_EmptyArray():
    v2a = ak.contents.emptyarray.EmptyArray()

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_NumpyArray_shape():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 1  # this one actually *should* touch


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_RegularArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_RegularArray_NumpyArray_v2b():
    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
        0,
        zeros_length=10,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_RecordArray_NumpyArray():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2c = ak.contents.recordarray.RecordArray([], [], 10)

    layout, report = ak.typetracer.typetracer_with_report(v2c.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2d = ak.contents.recordarray.RecordArray([], None, 10)

    layout, report = ak.typetracer.typetracer_with_report(v2d.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_IndexedArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2c.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2d.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_UnmaskedArray_NumpyArray():
    v2a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_UnionArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.from_iter(["1", "2", "3"], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([999.0, 0.0, 1.1, 2.2, 3.3]),
            parameters={"some": "stuff", "other": [1, 2, "three"]},
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_nested_NumpyArray_shape():
    data = np.full((3, 3, 5), 999, dtype=np.int64)
    data[1:3] = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)

    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(data),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 1  # this one actually *should* touch


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.regulararray.RegularArray(
            ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
            0,
            zeros_length=11,
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


@pytest.mark.parametrize("flatlist_as_rvec", [False, True])
def test_nested_ListOffsetArray_NumpyArray(flatlist_as_rvec):
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.listoffsetarray.ListOffsetArray(
            ak.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak.contents.numpyarray.NumpyArray(
                np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
            ),
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2c = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], [], 11),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2c.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

    v2d = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], None, 11),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2d.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_nested_IndexedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedarray.IndexedArray(
            ak.index.Index(np.array([999, 2, 2, 0, 1, 4, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_nested_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedoptionarray.IndexedOptionArray(
            ak.index.Index(np.array([999, 2, 2, -1, 1, -1, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2b.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2c.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form
    assert len(report.shape_touched) == 0

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

    layout, report = ak.typetracer.typetracer_with_report(v2d.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_nested_UnmaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.unmaskedarray.UnmaskedArray(
            ak.contents.numpyarray.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


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

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_ListArray_strings():
    v2a = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_RegularArray_strings():
    v2a = ak.operations.to_regular(
        ak.operations.from_iter(["onexx", "twoxx", "three", "fourx", "fivex"]),
        axis=1,
        highlevel=False,
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_NumpyArray_iterator():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_NumpyArray_iterator2():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0


def test_NumpyArray_riterator():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    layout, report = ak.typetracer.typetracer_with_report(v2a.form_with_key())
    assert len(report.shape_touched) == 0
    tmp = layout.form  # noqa: F841
    assert len(report.shape_touched) == 0
