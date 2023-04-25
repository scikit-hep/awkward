# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_from_iter():
    array = ak.from_iter(
        [
            1,
            2,
            "hi",
            [3, 4, {"x": 4}, None],
            [
                1j,
                [
                    3,
                    4,
                    (
                        False,
                        True,
                    ),
                ],
            ],
        ]
    )[:0]
    array_round_trip = ak.forms.from_type(array.type.content).length_zero_array()
    assert array_round_trip.type.is_equal_to(array.type)


def test_regular():
    array = ak.to_regular([[1, 2, 3]])[:0]
    array_round_trip = ak.forms.from_type(array.type.content).length_zero_array()
    assert array_round_trip.type.is_equal_to(array.type)
