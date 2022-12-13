# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    array = ak.Array(
        [[1.1, 2.2], [], [3.3, float("nan")], [float("-inf"), float("inf"), 7.7]]
    )

    assert to_list(
        ak.operations.nan_to_num(
            array, nan=999, posinf=float("-inf"), neginf=float("inf")
        )
    ) == [[1.1, 2.2], [], [3.3, 999.0], [float("inf"), float("-inf"), 7.7]]

    assert to_list(
        ak.operations.nan_to_num(
            array,
            nan=[[-1, -2], [], [-3, -4], [-5, -6, -7]],
            posinf=float("-inf"),
            neginf=float("inf"),
        )
    ) == [[1.1, 2.2], [], [3.3, -4], [float("inf"), float("-inf"), 7.7]]

    assert to_list(
        ak.operations.nan_to_num(
            array,
            nan=[[-1, -2], [], [-3, -4], [-5, -6, -7]],
            posinf=[[1, 2], [], [3, 4], [5, 6, 7]],
            neginf=float("inf"),
        )
    ) == [[1.1, 2.2], [], [3.3, -4], [float("inf"), 6.0, 7.7]]

    assert to_list(
        ak.operations.nan_to_num(
            array,
            nan=[[-1, -2], [], [-3, -4], [-5, -6, -7]],
            posinf=[[1, 2], [], [3, 4], [5, 6, 7]],
            neginf=[[10, 20], [], [30, 40], [50, 60, 70]],
        )
    ) == [[1.1, 2.2], [], [3.3, -4], [50.0, 6.0, 7.7]]

    assert to_list(
        ak.operations.nan_to_num(
            array,
            nan=[[-1, -2], [], [-3, -4], [-5, -6, -7]],
            posinf=float("-inf"),
            neginf=[[10, 20], [], [30, 40], [50, 60, 70]],
        )
    ) == [[1.1, 2.2], [], [3.3, -4], [50.0, float("-inf"), 7.7]]

    assert to_list(
        ak.operations.nan_to_num(
            array,
            nan=999,
            posinf=float("-inf"),
            neginf=[[10, 20], [], [30, 40], [50, 60, 70]],
        )
    ) == [[1.1, 2.2], [], [3.3, 999.0], [50.0, float("-inf"), 7.7]]
