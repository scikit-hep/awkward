# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
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

    def check(layout):
        if hasattr(layout, "contents"):
            for x in layout.contents:
                check(x)
        elif hasattr(layout, "content"):
            check(layout.content)
        else:
            assert layout.length <= 3

    for _ in range(5):
        check(cat.layout)

        cat["another", "w"] = three.x
