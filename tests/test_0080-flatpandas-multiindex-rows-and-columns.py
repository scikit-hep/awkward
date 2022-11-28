# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import numpy as np  # noqa: F401
import pytest
import setuptools

import awkward as ak

pandas = pytest.importorskip("pandas")


@pytest.mark.skipif(
    setuptools.extern.packaging.version.parse(pandas.__version__)
    < setuptools.extern.packaging.version.parse("1.0"),
    reason="Test Pandas in 1.0+ because they had to fix their JSON format.",
)
def test():
    def key(n):
        if n in ("values", "x", "y"):
            return n
        else:
            return tuple(eval(n.replace("nan", "None").replace("null", "None")))

    def regularize(data):
        if isinstance(data, dict):
            return {key(n): regularize(x) for n, x in data.items()}
        else:
            return data

    array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, None, 8.8, 9.9]])
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        "values": {
            (0, 0): 0.0,
            (0, 1): 1.1,
            (0, 2): 2.2,
            (2, 0): 3.3,
            (2, 1): 4.4,
            (3, 0): 5.5,
            (4, 0): 6.6,
            (4, 1): None,
            (4, 2): 8.8,
            (4, 3): 9.9,
        }
    }

    array = ak.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [[5.5]], [[6.6, None, 8.8, 9.9]]]
    )
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        "values": {
            (0, 0, 0): 0.0,
            (0, 0, 1): 1.1,
            (0, 0, 2): 2.2,
            (0, 2, 0): 3.3,
            (0, 2, 1): 4.4,
            (1, 0, 0): 5.5,
            (2, 0, 0): 6.6,
            (2, 0, 1): None,
            (2, 0, 2): 8.8,
            (2, 0, 3): 9.9,
        }
    }

    array = ak.Array(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            None,
            [[], None, [6.6, None, 8.8, 9.9]],
        ]
    )
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        "values": {
            (0, 0, 0): 0.0,
            (0, 0, 1): 1.1,
            (0, 0, 2): 2.2,
            (0, 2, 0): 3.3,
            (0, 2, 1): 4.4,
            (2, 0, 0): 5.5,
            (4, 2, 0): 6.6,
            (4, 2, 1): None,
            (4, 2, 2): 8.8,
            (4, 2, 3): 9.9,
        }
    }

    array = ak.Array(
        [
            [
                [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                [],
                [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}],
            ],
            [],
            [[{"x": 5.5, "y": [5, 5, 5, 5, 5]}]],
        ]
    )
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        "x": {
            (0, 0, 1, 0): 1.1,
            (0, 0, 2, 0): 2.2,
            (0, 0, 2, 1): 2.2,
            (0, 2, 0, 0): 3.3,
            (0, 2, 0, 1): 3.3,
            (0, 2, 0, 2): 3.3,
            (0, 2, 1, 0): 4.4,
            (0, 2, 1, 1): 4.4,
            (0, 2, 1, 2): 4.4,
            (0, 2, 1, 3): 4.4,
            (2, 0, 0, 0): 5.5,
            (2, 0, 0, 1): 5.5,
            (2, 0, 0, 2): 5.5,
            (2, 0, 0, 3): 5.5,
            (2, 0, 0, 4): 5.5,
        },
        "y": {
            (0, 0, 1, 0): 1,
            (0, 0, 2, 0): 2,
            (0, 0, 2, 1): 2,
            (0, 2, 0, 0): 3,
            (0, 2, 0, 1): 3,
            (0, 2, 0, 2): 3,
            (0, 2, 1, 0): 4,
            (0, 2, 1, 1): 4,
            (0, 2, 1, 2): 4,
            (0, 2, 1, 3): 4,
            (2, 0, 0, 0): 5,
            (2, 0, 0, 1): 5,
            (2, 0, 0, 2): 5,
            (2, 0, 0, 3): 5,
            (2, 0, 0, 4): 5,
        },
    }

    assert regularize(
        json.loads(ak.operations.to_dataframe(array, how="outer").to_json())
    ) == {
        "x": {
            (0, 0, 0, None): 0.0,
            (0, 0, 1, 0.0): 1.1,
            (0, 0, 2, 0.0): 2.2,
            (0, 0, 2, 1.0): 2.2,
            (0, 2, 0, 0.0): 3.3,
            (0, 2, 0, 1.0): 3.3,
            (0, 2, 0, 2.0): 3.3,
            (0, 2, 1, 0.0): 4.4,
            (0, 2, 1, 1.0): 4.4,
            (0, 2, 1, 2.0): 4.4,
            (0, 2, 1, 3.0): 4.4,
            (2, 0, 0, 0.0): 5.5,
            (2, 0, 0, 1.0): 5.5,
            (2, 0, 0, 2.0): 5.5,
            (2, 0, 0, 3.0): 5.5,
            (2, 0, 0, 4.0): 5.5,
        },
        "y": {
            (0, 0, 0, None): None,
            (0, 0, 1, 0.0): 1.0,
            (0, 0, 2, 0.0): 2.0,
            (0, 0, 2, 1.0): 2.0,
            (0, 2, 0, 0.0): 3.0,
            (0, 2, 0, 1.0): 3.0,
            (0, 2, 0, 2.0): 3.0,
            (0, 2, 1, 0.0): 4.0,
            (0, 2, 1, 1.0): 4.0,
            (0, 2, 1, 2.0): 4.0,
            (0, 2, 1, 3.0): 4.0,
            (2, 0, 0, 0.0): 5.0,
            (2, 0, 0, 1.0): 5.0,
            (2, 0, 0, 2.0): 5.0,
            (2, 0, 0, 3.0): 5.0,
            (2, 0, 0, 4.0): 5.0,
        },
    }

    array = ak.Array(
        [
            [
                [{"x": 0.0, "y": 0}, {"x": 1.1, "y": 1}, {"x": 2.2, "y": 2}],
                [],
                [{"x": 3.3, "y": 3}, {"x": 4.4, "y": 4}],
            ],
            [],
            [[{"x": 5.5, "y": 5}]],
        ]
    )
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        "x": {
            (0, 0, 0): 0.0,
            (0, 0, 1): 1.1,
            (0, 0, 2): 2.2,
            (0, 2, 0): 3.3,
            (0, 2, 1): 4.4,
            (2, 0, 0): 5.5,
        },
        "y": {
            (0, 0, 0): 0,
            (0, 0, 1): 1,
            (0, 0, 2): 2,
            (0, 2, 0): 3,
            (0, 2, 1): 4,
            (2, 0, 0): 5,
        },
    }

    array = ak.Array(
        [
            [
                [
                    {"x": 0.0, "y": {"z": 0}},
                    {"x": 1.1, "y": {"z": 1}},
                    {"x": 2.2, "y": {"z": 2}},
                ],
                [],
                [{"x": 3.3, "y": {"z": 3}}, {"x": 4.4, "y": {"z": 4}}],
            ],
            [],
            [[{"x": 5.5, "y": {"z": 5}}]],
        ]
    )
    assert regularize(json.loads(ak.operations.to_dataframe(array).to_json())) == {
        ("x", ""): {
            (0, 0, 0): 0.0,
            (0, 0, 1): 1.1,
            (0, 0, 2): 2.2,
            (0, 2, 0): 3.3,
            (0, 2, 1): 4.4,
            (2, 0, 0): 5.5,
        },
        ("y", "z"): {
            (0, 0, 0): 0,
            (0, 0, 1): 1,
            (0, 0, 2): 2,
            (0, 2, 0): 3,
            (0, 2, 1): 4,
            (2, 0, 0): 5,
        },
    }

    one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    two = ak.Array([[100, 200], [300], [400, 500]])
    assert [
        regularize(json.loads(x.to_json()))
        for x in ak.operations.to_dataframe({"x": one, "y": two}, how=None)
    ] == [
        {"x": {(0, 0): 1.1, (0, 1): 2.2, (0, 2): 3.3, (2, 0): 4.4, (2, 1): 5.5}},
        {"y": {(0, 0): 100, (0, 1): 200, (1, 0): 300, (2, 0): 400, (2, 1): 500}},
    ]
