# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np

import awkward as ak


def test():
    def _reduce_max_masked(array, mask):
        assert mask
        j = ak.from_regular(
            ak.argmax(array["1"], axis=1, keepdims=True, mask_identity=True)
        )
        return ak.flatten(array[j], axis=1)

    behavior = {}
    behavior[ak.max, "pair"] = _reduce_max_masked

    content = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([2]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 6]),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
                    ak.contents.NumpyArray(np.linspace(0, 10, num=6, dtype=np.float64)),
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

    content = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([2]),
        ak.contents.ListArray(
            ak.index.Index64([0, 3]),
            ak.index.Index64([3, 6]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64([0, 2, 4, 1, 3, 5]),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([0, 3, 1, 4, 2, 5], dtype=np.int64)
                        ),
                        ak.contents.NumpyArray(
                            np.array([0, 6, 2, 8, 4, 10], dtype=np.float64)
                        ),
                    ],
                    fields=None,
                    parameters={"__record__": "pair"},
                ),
            ),
        ),
    )
    assert result.is_equal_to(content)
