# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    one_a = ak.Array([{"x": 1, "y": 2}], with_name="T")
    one_b = ak.Array([{"x": 1, "y": 2}], with_name="T")
    two_a = ak.Array([{"x": 1, "z": 3}], with_name="T")
    two_b = ak.Array([{"x": 1, "z": 3}], with_name="T")
    three = ak.Array([{"x": 4}, {"x": 4}], with_name="T")

    first = ak.zip({"a": one_a, "b": one_b})
    second = ak.zip({"a": two_a, "b": two_b})

    cat = ak.concatenate([first, second], axis=0)

    cat["another"] = three

    def check(layout):
        if hasattr(layout, "contents"):
            for x in layout.contents:
                check(x)
        elif hasattr(layout, "content"):
            check(layout.content)
        else:
            assert layout.length <= 2

    for _ in range(5):
        check(cat.layout)

        cat["another", "w"] = three.x
