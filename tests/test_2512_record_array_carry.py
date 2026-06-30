# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def _reduce_max_masked(array, mask):
    assert mask
    j = ak.from_regular(
        ak.argmax(array["1"], axis=1, keepdims=True, mask_identity=True)
    )
    return ak.flatten(array[j], axis=1)


behavior = {}
behavior[ak.max, "pair"] = _reduce_max_masked


def test_axis_0():
    content = ak.contents.ListArray(
        ak.index.Index64([0, 2]),
        ak.index.Index64([2, 4]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6, 9, 11]),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(11, dtype=np.int64)),
                    ak.contents.NumpyArray(
                        np.array(
                            [
                                0.0,
                                2.0,
                                4.0,
                                6.0,
                                8.0,
                                10.0,
                                1.0,
                                14.0,
                                16.0,
                                18.0,
                                20.0,
                            ],
                            dtype=np.float64,
                        )
                    ),
                ],
                fields=None,
                parameters={"__record__": "pair"},
            ),
        ),
    )

    result = ak.max(
        content,
        axis=0,
        keepdims=True,
        mask_identity=True,
        behavior=behavior,
        highlevel=False,
    )

    # The offsets-pipeline migration of `awkward_ListOffsetArray_reduce_nonlocal_preparenext_64`
    # iterates `(outer_bin, col, row)` rather than the parents-era `(round, row)`,
    # producing a structurally different — but semantically equivalent — layout
    # for the carried inner records: bin-major (locally column-major within each
    # outer bin) instead of globally column-major. The IndexedOptionArray.index
    # is therefore monotonic, and the underlying RecordArray contents are
    # interleaved per outer bin rather than across all rows. Both representations
    # describe the same set of picked records; only the structural permutation
    # differs.
    expected_result = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([2]),
        ak.contents.ListArray(
            ak.index.Index64([0, 3]),
            ak.index.Index64([3, 6]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64([1, 3, 5, 7, 9, 10]),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5], dtype=np.int64)
                        ),
                        ak.contents.NumpyArray(
                            np.array(
                                [0, 1, 2, 14, 4, 16, 6, 18, 8, 20, 10],
                                dtype=np.float64,
                            )
                        ),
                    ],
                    fields=None,
                    parameters={"__record__": "pair"},
                ),
            ),
        ),
    )
    assert result.is_equal_to(expected_result)


def test_axis_1():
    content = ak.contents.ListArray(
        ak.index.Index64([0, 2]),
        ak.index.Index64([2, 4]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6, 9, 11]),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(11, dtype=np.int64)),
                    ak.contents.NumpyArray(
                        np.array(
                            [
                                0.0,
                                2.0,
                                4.0,
                                6.0,
                                8.0,
                                10.0,
                                1.0,
                                14.0,
                                16.0,
                                18.0,
                                20.0,
                            ],
                            dtype=np.float64,
                        )
                    ),
                ],
                fields=None,
                parameters={"__record__": "pair"},
            ),
        ),
    )

    result = ak.max(
        content,
        axis=1,
        keepdims=True,
        mask_identity=True,
        behavior=behavior,
        highlevel=False,
    )

    # See the note in test_axis_0 about the structural difference between the
    # parents-era and offsets-pipeline carry layouts.
    expected_result = ak.contents.RegularArray(
        ak.contents.ListArray(
            ak.index.Index64([0, 3]),
            ak.index.Index64([3, 6]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64([1, 3, 5, 7, 9, 10]),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8], dtype=np.int64)
                        ),
                        ak.contents.NumpyArray(
                            np.array(
                                [0, 6, 2, 8, 4, 10, 1, 18, 14, 20, 16],
                                dtype=np.float64,
                            )
                        ),
                    ],
                    fields=None,
                    parameters={"__record__": "pair"},
                ),
            ),
        ),
        size=1,
    )
    assert result.is_equal_to(expected_result)
