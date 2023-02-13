# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    aa = ak.contents.NumpyArray(np.frombuffer(b"hellothere", "u1"))
    b = aa.to_backend_array(allow_missing=False)
    assert b.tolist() == [104, 101, 108, 108, 111, 116, 104, 101, 114, 101]
    assert b.dtype == np.dtype(np.uint8)

    c = ak.contents.NumpyArray(np.array([0, 1577836800], dtype="datetime64[s]"))
    assert [
        d.isoformat() for d in c.to_backend_array(allow_missing=False).tolist()
    ] == [
        "1970-01-01T00:00:00",
        "2020-01-01T00:00:00",
    ]

    recordarray = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))],
        fields=["one"],
    )
    assert recordarray.to_backend_array(allow_missing=False).tolist() == [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
    ]
