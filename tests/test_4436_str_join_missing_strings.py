# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def test_join_with_missing_strings():
    array = ak.Array([["this", "that"], ["foo", None, "baz"], [None], []])
    result = ak.str.join(array, "-")
    assert result.tolist() == ["this-that", None, None, ""]
    assert str(result.type) == "4 * ?string"
    assert (
        result.layout.form
        == ak.str.join(ak.to_backend(array, "typetracer"), "-").layout.form
    )

    array = ak.Array([[b"this", b"that"], [b"foo", None]])
    assert ak.str.join(array, b"-").tolist() == [b"this-that", None]


def test_join_with_missing_separator():
    array = ak.Array([["this", "that"], ["foo", "bar"], ["baz"]])
    separator = ak.Array(["-", None, "+"])
    result = ak.str.join(array, separator)
    assert result.tolist() == ["this-that", None, "baz"]
    assert (
        result.layout.form
        == ak.str.join(ak.to_backend(array, "typetracer"), separator).layout.form
    )
