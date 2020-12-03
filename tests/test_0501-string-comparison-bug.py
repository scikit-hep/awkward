# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    one = ak.Array(["uno", "dos", "tres"])
    two = ak.Array(["un", "deux", "trois", "quatre"])
    three = ak.Array(["onay", "ootay", "eethray"])
    merged = ak.concatenate([one, two, three])
    assert ak.to_list(merged) == [
        "uno",
        "dos",
        "tres",
        "un",
        "deux",
        "trois",
        "quatre",
        "onay",
        "ootay",
        "eethray",
    ]
    assert ak.to_list(merged == "uno") == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert ak.to_list(one == np.array(["UNO", "dos", "tres"])) == [False, True, True]
    assert ak.to_list(
        merged
        == np.array(
            [
                "UNO",
                "dos",
                "tres",
                "one",
                "two",
                "three",
                "quatre",
                "onay",
                "two",
                "three",
            ]
        )
    ) == [False, True, True, False, False, False, True, True, False, False]


def test_fromnumpy():
    assert ak.to_list(ak.from_numpy(np.array(["uno", "dos", "tres", "quatro"]))) == [
        "uno",
        "dos",
        "tres",
        "quatro",
    ]
    assert ak.to_list(
        ak.from_numpy(np.array([["uno", "dos"], ["tres", "quatro"]]))
    ) == [["uno", "dos"], ["tres", "quatro"]]
    assert ak.to_list(
        ak.from_numpy(np.array([["uno", "dos"], ["tres", "quatro"]]), regulararray=True)
    ) == [["uno", "dos"], ["tres", "quatro"]]
