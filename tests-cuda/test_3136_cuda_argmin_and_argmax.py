from __future__ import annotations

import awkward as ak

to_list = ak.operations.to_list


def test_argmin_argmax_axis_None():
    array = ak.highlevel.Array(
        [
            [
                [2022, 2023, 2025],
                [],
                [2027, 2011],
                [2013],
            ],
            [],
            [[2017, 2019], [2023]],
        ],
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.argmin(cuda_array) == 4
    assert ak.operations.argmax(cuda_array) == 3


def test():
    array = ak.highlevel.Array([1, 2, 3, None, 4])

    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.argmax(cuda_array) == 4
