# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from pathlib import Path

import numpy as np
import pytest

import awkward as ak

samples_path = Path(__file__).parent / "samples"

to_list = ak.operations.to_list


def test_unfinished_fragment_exception():
    # read unfinished json fragments
    strs0 = """{"one": 1, "two": 2.2,"""
    with pytest.raises(ValueError):
        ak.operations.from_json(strs0)

    strs1 = """{"one": 1,
        "two": 2.2,"""
    with pytest.raises(ValueError):
        ak.operations.from_json(strs1)

    strs2 = """{"one": 1,
        "two": 2.2,
        """
    with pytest.raises(ValueError):
        ak.operations.from_json(strs2)

    strs3 = """{"one": 1, "two": 2.2, "three": "THREE"}
        {"one": 10, "two": 22,"""
    with pytest.raises(ValueError):
        ak.operations.from_json(strs3)

    strs4 = """{"one": 1, "two": 2.2, "three": "THREE"}
        {"one": 10, "two": 22,
        """
    with pytest.raises(ValueError):
        ak.operations.from_json(strs4)

    strs5 = """["one", "two","""
    with pytest.raises(ValueError):
        ak.operations.from_json(strs5)

    strs6 = """["one",
        "two","""
    with pytest.raises(ValueError):
        ak.operations.from_json(strs6)

    strs7 = """["one",
        "two",
        """
    with pytest.raises(ValueError):
        ak.operations.from_json(strs7)


def test_two_arrays():
    str = """{"one": 1, "two": 2.2}{"one": 10, "two": 22}"""
    with pytest.raises(ValueError):
        ak.operations.from_json(str)

    str = """{"one": 1, "two": 2.2}     {"one": 10, "two": 22}"""
    with pytest.raises(ValueError):
        ak.operations.from_json(str)

    str = """{"one": 1, \t "two": 2.2}{"one": 10, "two": 22}"""
    with pytest.raises(ValueError):
        ak.operations.from_json(str)

    str = """{"one": 1, "two": 2.2}  \t   {"one": 10, "two": 22}"""
    with pytest.raises(ValueError):
        ak.operations.from_json(str)

    str = """{"one": 1, "two": 2.2}\n{"one": 10, "two": 22}"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}\n\r{"one": 10, "two": 22}"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}     \n     {"one": 10, "two": 22}"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}     \n\r     {"one": 10, "two": 22}"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}\n{"one": 10, "two": 22}\n"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}\n\r{"one": 10, "two": 22}\n\r"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """["one", "two"]\n["uno", "dos"]"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [["one", "two"], ["uno", "dos"]]

    str = """["one", "two"]\n\r["uno", "dos"]"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [["one", "two"], ["uno", "dos"]]

    str = """["one", "two"]  \n   ["uno", "dos"]"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [["one", "two"], ["uno", "dos"]]

    str = """["one", "two"]  \n\r   ["uno", "dos"]"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [["one", "two"], ["uno", "dos"]]

    str = '"one"\n"two"'
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == ["one", "two"]

    str = '"one"\n\r"two"'
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == ["one", "two"]

    str = '"one"  \n   "two"'
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == ["one", "two"]

    array = ak.operations.from_json(
        samples_path / "test-two-arrays.json", line_delimited=True
    )
    assert array.to_list() == [
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        {"one": 1, "two": 2.2},
        {"one": 10, "two": 22.0},
        ["one", "two"],
        ["uno", "dos"],
        ["one", "two"],
        ["uno", "dos"],
        ["one", "two"],
        ["uno", "dos"],
        ["one", "two"],
        ["uno", "dos"],
        ["one", "two"],
        ["uno", "dos"],
        ["one", "two"],
        ["uno", "dos"],
        "one",
        "two",
        "one",
        "two",
        "one",
        "two",
        "one",
        "two",
        "one",
        "two",
        "one",
        "two",
    ]


def test_blanc_lines():
    str = """{"one": 1, "two": 2.2}

    {"one": 10, "two": 22}"""
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """{"one": 1, "two": 2.2}

    {"one": 10, "two": 22}
    """
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [{"one": 1, "two": 2.2}, {"one": 10, "two": 22.0}]

    str = """ 1
    2

    3   """
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [1, 2, 3]

    str = """
        1
        2

        3
        """
    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [1, 2, 3]


def test_tostring():
    # write a json string from an array built from
    # multiple json fragments from a string
    str = """{"x": 1.1, "y": []}
             {"x": 2.2, "y": [1]}
             {"x": 3.3, "y": [1, 2]}
             {"x": 4.4, "y": [1, 2, 3]}
             {"x": 5.5, "y": [1, 2, 3, 4]}
             {"x": 6.6, "y": [1, 2, 3, 4, 5]}"""

    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [
        {"x": 1.1, "y": []},
        {"x": 2.2, "y": [1]},
        {"x": 3.3, "y": [1, 2]},
        {"x": 4.4, "y": [1, 2, 3]},
        {"x": 5.5, "y": [1, 2, 3, 4]},
        {"x": 6.6, "y": [1, 2, 3, 4, 5]},
    ]

    assert (
        ak.operations.to_json(array)
        == '[{"x":1.1,"y":[]},{"x":2.2,"y":[1]},{"x":3.3,"y":[1,2]},{"x":4.4,"y":[1,2,3]},{"x":5.5,"y":[1,2,3,4]},{"x":6.6,"y":[1,2,3,4,5]}]'
    )


def test_fromstring():
    # read multiple json fragments from a string
    str = """{"x": 1.1, "y": []}
             {"x": 2.2, "y": [1]}
             {"x": 3.3, "y": [1, 2]}
             {"x": 4.4, "y": [1, 2, 3]}
             {"x": 5.5, "y": [1, 2, 3, 4]}
             {"x": 6.6, "y": [1, 2, 3, 4, 5]}"""

    array = ak.operations.from_json(str, line_delimited=True)
    assert array.to_list() == [
        {"x": 1.1, "y": []},
        {"x": 2.2, "y": [1]},
        {"x": 3.3, "y": [1, 2]},
        {"x": 4.4, "y": [1, 2, 3]},
        {"x": 5.5, "y": [1, 2, 3, 4]},
        {"x": 6.6, "y": [1, 2, 3, 4, 5]},
    ]


def test_array_tojson():
    # convert float 'nan' and 'inf' to user-defined strings
    array = ak.contents.NumpyArray(
        np.array(
            [[float("nan"), float("nan"), 1.1], [float("inf"), 3.3, float("-inf")]]
        )
    )

    assert (
        ak.operations.to_json(
            array, nan_string="NaN", posinf_string="inf", neginf_string="-inf"
        )
        == '[["NaN","NaN",1.1],["inf",3.3,"-inf"]]'
    )

    array2 = ak.highlevel.Array([[0, 2], None, None, None, "NaN", "NaN"])
    assert (
        ak.operations.to_json(array2, nan_string="NaN")
        == '[[0,2],null,null,null,"NaN","NaN"]'
    )


def test_fromfile():
    # read multiple json fragments from a json file
    array = ak.operations.from_json(
        samples_path / "test-record-array.json", line_delimited=True
    )
    assert array.to_list() == [
        {"x": 1.1, "y": []},
        {"x": 2.2, "y": [1]},
        {"x": 3.3, "y": [1, 2]},
        {"x": 4.4, "y": [1, 2, 3]},
        {"x": 5.5, "y": [1, 2, 3, 4]},
        {"x": 6.6, "y": [1, 2, 3, 4, 5]},
    ]

    # read json file containing 'nan' and 'inf' user-defined strings
    # and replace 'nan' and 'inf' strings with floats
    array = ak.operations.from_json(
        samples_path / "test.json",
        posinf_string="inf",
        neginf_string="-inf",
    )

    assert array.to_list() == [
        1.1,
        2.2,
        3.3,
        float("inf"),
        float("-inf"),
        [4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11],
        [12.12, 13.13, 14.14, 15.15, 16.16, 17.17],
        [
            [
                [18.18, 19.19, 20.2, 21.21, 22.22],
                [
                    23.23,
                    24.24,
                    25.25,
                    26.26,
                    27.27,
                    28.28,
                    29.29,
                    30.3,
                    31.31,
                    32.32,
                    33.33,
                    34.34,
                    35.35,
                    36.36,
                    37.37,
                ],
                [38.38],
                [39.39, 40.4, "NaN", "NaN", 41.41, 42.42, 43.43],
            ],
            [
                [44.44, 45.45, 46.46, 47.47, 48.48],
                [
                    49.49,
                    50.5,
                    51.51,
                    52.52,
                    53.53,
                    54.54,
                    55.55,
                    56.56,
                    57.57,
                    58.58,
                    59.59,
                    60.6,
                    61.61,
                    62.62,
                    63.63,
                ],
                [64.64],
                [65.65, 66.66, "NaN", "NaN", 67.67, 68.68, 69.69],
            ],
        ],
    ]

    # read json file containing 'nan' and 'inf' user-defined strings
    array = ak.operations.from_json(samples_path / "test.json")

    assert array.to_list() == [
        1.1,
        2.2,
        3.3,
        "inf",
        "-inf",
        [4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11],
        [12.12, 13.13, 14.14, 15.15, 16.16, 17.17],
        [
            [
                [18.18, 19.19, 20.2, 21.21, 22.22],
                [
                    23.23,
                    24.24,
                    25.25,
                    26.26,
                    27.27,
                    28.28,
                    29.29,
                    30.3,
                    31.31,
                    32.32,
                    33.33,
                    34.34,
                    35.35,
                    36.36,
                    37.37,
                ],
                [38.38],
                [39.39, 40.4, "NaN", "NaN", 41.41, 42.42, 43.43],
            ],
            [
                [44.44, 45.45, 46.46, 47.47, 48.48],
                [
                    49.49,
                    50.5,
                    51.51,
                    52.52,
                    53.53,
                    54.54,
                    55.55,
                    56.56,
                    57.57,
                    58.58,
                    59.59,
                    60.6,
                    61.61,
                    62.62,
                    63.63,
                ],
                [64.64],
                [65.65, 66.66, "NaN", "NaN", 67.67, 68.68, 69.69],
            ],
        ],
    ]

    # read json file containing 'nan' and 'inf' user-defined strings
    # and replace 'nan' and 'inf' strings with a predefined 'None' string
    array = ak.operations.from_json(
        samples_path / "test.json",
        posinf_string="inf",
        neginf_string="-inf",
        nan_string="NaN",
    )

    def fix(obj):
        if isinstance(obj, list):
            return [fix(x) for x in obj]
        elif np.isnan(obj):
            return "COMPARE-NAN"
        else:
            return obj

    assert fix(array.to_list()) == fix(
        [
            1.1,
            2.2,
            3.3,
            float("inf"),
            float("-inf"),
            [4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11],
            [12.12, 13.13, 14.14, 15.15, 16.16, 17.17],
            [
                [
                    [18.18, 19.19, 20.2, 21.21, 22.22],
                    [
                        23.23,
                        24.24,
                        25.25,
                        26.26,
                        27.27,
                        28.28,
                        29.29,
                        30.3,
                        31.31,
                        32.32,
                        33.33,
                        34.34,
                        35.35,
                        36.36,
                        37.37,
                    ],
                    [38.38],
                    [39.39, 40.4, float("nan"), float("nan"), 41.41, 42.42, 43.43],
                ],
                [
                    [44.44, 45.45, 46.46, 47.47, 48.48],
                    [
                        49.49,
                        50.5,
                        51.51,
                        52.52,
                        53.53,
                        54.54,
                        55.55,
                        56.56,
                        57.57,
                        58.58,
                        59.59,
                        60.6,
                        61.61,
                        62.62,
                        63.63,
                    ],
                    [64.64],
                    [65.65, 66.66, float("nan"), float("nan"), 67.67, 68.68, 69.69],
                ],
            ],
        ]
    )

    # read json file containing multiple definitions of 'nan' and 'inf'
    # user-defined strings
    # replace can only work for one string definition
    array = ak.operations.from_json(
        samples_path / "test-nan-inf.json",
        posinf_string="Infinity",
        nan_string="None at all",
    )

    assert array.to_list() == [
        1.1,
        2.2,
        3.3,
        "inf",
        "-inf",
        [4.4, float("inf"), 6.6, 7.7, 8.8, "NaN", 10.1, 11.11],
        [12.12, 13.13, 14.14, 15.15, 16.16, 17.17],
        [
            [
                [18.18, 19.19, 20.2, 21.21, 22.22],
                [
                    23.23,
                    24.24,
                    25.25,
                    26.26,
                    27.27,
                    28.28,
                    29.29,
                    30.3,
                    31.31,
                    32.32,
                    33.33,
                    34.34,
                    35.35,
                    36.36,
                    37.37,
                ],
                [38.38],
                [39.39, 40.4, "NaN", "NaN", 41.41, 42.42, 43.43],
            ],
            [
                [44.44, 45.45, 46.46, 47.47, 48.48],
                [
                    49.49,
                    50.5,
                    51.51,
                    52.52,
                    53.53,
                    54.54,
                    55.55,
                    56.56,
                    57.57,
                    58.58,
                    59.59,
                    60.6,
                    61.61,
                    62.62,
                    63.63,
                ],
                [64.64],
                [65.65, 66.66, "NaN", "NaN", 67.67, 68.68, 69.69],
            ],
        ],
    ]


def test_three():
    array = ak.operations.from_json('["one", "two"] \n ["three"]', line_delimited=True)
    assert array.to_list() == [["one", "two"], ["three"]]


def test_jpivarski():
    assert to_list(ak.operations.from_json('{"x": 1, "y": [1, 2, 3]}')) == {
        "x": 1,
        "y": [1, 2, 3],
    }

    with pytest.raises(ValueError):
        ak.operations.from_json('{"x": 1, "y": [1, 2, 3]} {"x": 2, "y": []}')

    with pytest.raises(ValueError):
        ak.operations.from_json('{"x": 1, "y": [1, 2, 3]} 123')

    with pytest.raises(ValueError):
        ak.operations.from_json('{"x": 1, "y": [1, 2, 3]} [1, 2, 3, 4, 5]')

    assert ak.operations.from_json("123") == 123

    with pytest.raises(ValueError):
        ak.operations.from_json("123 456")

    with pytest.raises(ValueError):
        ak.operations.from_json('123 {"x": 1, "y": [1, 2, 3]}')

    assert ak.operations.from_json("null") is None

    with pytest.raises(ValueError):
        ak.operations.from_json("null 123")

    with pytest.raises(ValueError):
        ak.operations.from_json("123 null")
