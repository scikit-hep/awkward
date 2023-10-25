# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import numpy as np

import awkward as ak


def test_strings():
    assert ak.to_layout("hello").is_equal_to(
        ak.contents.NumpyArray(
            np.array([104, 101, 108, 108, 111], dtype=np.uint8),
            parameters={"__array__": "char"},
        )
    )
    assert ak.to_layout("hello", string_policy="promote").is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 5]),
            ak.contents.NumpyArray(
                np.array([104, 101, 108, 108, 111], dtype=np.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )
    )
    assert ak.to_layout("hello", string_policy="pass-through") == "hello"
