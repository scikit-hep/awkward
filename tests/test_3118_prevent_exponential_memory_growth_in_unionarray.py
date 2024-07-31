# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def check(layout, assert_length):
    if hasattr(layout, "contents"):
        for x in layout.contents:
            check(x, assert_length)
    elif hasattr(layout, "content"):
        check(layout.content, assert_length)
    else:
        assert layout.length <= assert_length


def test_2arrays():
    one_a = ak.Array([{"x": 1, "y": 2}], with_name="T")
    one_b = ak.Array([{"x": 1, "y": 2}], with_name="T")
    two_a = ak.Array([{"x": 1, "z": 3}], with_name="T")
    two_b = ak.Array([{"x": 1, "z": 3}], with_name="T")
    three = ak.Array([{"x": 4}, {"x": 4}], with_name="T")

    first = ak.zip({"a": one_a, "b": one_b})
    second = ak.zip({"a": two_a, "b": two_b})

    cat = ak.concatenate([first, second], axis=0)

    cat["another"] = three

    for _ in range(5):
        check(cat.layout, 2)

        cat["another", "w"] = three.x


def test_3arrays():
    zero_a = ak.Array([{"x": 1, "y": 1}], with_name="T")
    zero_b = ak.Array([{"x": 1, "v": 1}], with_name="T")
    one_a = ak.Array([{"x": 1, "y": 2}], with_name="T")
    one_b = ak.Array([{"x": 1, "y": 2}], with_name="T")
    two_a = ak.Array([{"x": 1, "z": 3}], with_name="T")
    two_b = ak.Array([{"x": 1, "z": 3}], with_name="T")
    three = ak.Array([{"x": 4}, {"x": 4}, {"x": 4}], with_name="T")

    zeroth = ak.zip({"a": zero_a, "b": zero_b})
    first = ak.zip({"a": one_a, "b": one_b})
    second = ak.zip({"a": two_a, "b": two_b})

    cat = ak.concatenate([zeroth, first, second], axis=0)

    cat["another"] = three

    for _ in range(5):
        check(cat.layout, 3)

        cat["another", "w"] = three.x
