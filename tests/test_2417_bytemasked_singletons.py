# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array(
        ak.contents.ByteMaskedArray(
            mask=ak.index.Index8([1, 1, 0, 0]),
            content=ak.contents.NumpyArray([3, 4, 2, 5]),
            valid_when=True,
        )
    )
    assert (
        ak.singletons(array, highlevel=False).form
        == ak.singletons(ak.to_backend(array, "typetracer"), highlevel=False).form
    )

    assert (
        ak.singletons(array, highlevel=False).form
        == ak.singletons(
            array.layout.to_typetracer(forget_length=True), highlevel=False
        ).form
    )
