# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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

    expected_result = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([2]),
        ak.contents.ListArray(
            ak.index.Index64([0, 3]),
            ak.index.Index64([3, 6]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64([1, 5, 9, 3, 7, 10]),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5], dtype=np.int64)
                        ),
                        ak.contents.NumpyArray(
                            np.array(
                                [0, 1, 6, 18, 2, 14, 8, 20, 4, 16, 10],
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

    expected_result = ak.contents.RegularArray(
        ak.contents.ListArray(
            ak.index.Index64([0, 3]),
            ak.index.Index64([3, 6]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64([1, 5, 9, 3, 7, 10]),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8], dtype=np.int64)
                        ),
                        ak.contents.NumpyArray(
                            np.array(
                                [0, 6, 1, 18, 2, 8, 14, 20, 4, 10, 16],
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
