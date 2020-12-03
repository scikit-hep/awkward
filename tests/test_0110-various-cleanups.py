# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_na_union():
    one = ak.from_iter([1, None, 3], highlevel=False)
    two = ak.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak.Array(
        ak.layout.UnionArray8_64(tags, index, [one, two]), check_valid=True
    )
    assert ak.to_list(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak.to_list(ak.is_none(array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]


class DummyRecord(ak.Record):
    def __repr__(self):
        return "<{0}>".format(self.x)


class DummyArray(ak.Array):
    def __repr__(self):
        return "<DummyArray {0}>".format(" ".join(repr(x) for x in self))


class DeepDummyArray(ak.Array):
    def __repr__(self):
        return "<DeepDummyArray {0}>".format(" ".join(repr(x) for x in self))


def test_behaviors():
    behavior = {}
    behavior["Dummy"] = DummyRecord
    behavior[".", "Dummy"] = DummyArray
    behavior["*", "Dummy"] = DeepDummyArray

    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    recordarray = ak.layout.RecordArray({"x": content})
    recordarray.setparameter("__record__", "Dummy")

    array = ak.Array(recordarray, behavior=behavior, check_valid=True)
    assert repr(array) == "<DummyArray <1.1> <2.2> <3.3> <4.4> <5.5>>"
    assert repr(array[0]) == "<1.1>"

    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, recordarray)

    array2 = ak.Array(listoffsetarray, behavior=behavior, check_valid=True)

    assert array2.layout.parameter("__record__") is None
    assert array2.layout.purelist_parameter("__record__") == "Dummy"

    assert (
        repr(array2)
        == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"
    )
    assert repr(array2[0]) == "<DummyArray <1.1> <2.2> <3.3>>"
    assert repr(array2[0, 0]) == "<1.1>"

    recordarray2 = ak.layout.RecordArray({"outer": listoffsetarray})

    array3 = ak.Array(recordarray2, behavior=behavior, check_valid=True)
    assert type(array3) is ak.Array
    assert type(array3["outer"]) is DeepDummyArray
    assert (
        repr(array3["outer"])
        == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"
    )


def test_flatten():
    assert ak.to_list(
        ak.flatten(
            ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True), axis=1
        )
    ) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_string_equal():
    trials = [
        (
            ["one", "two", "", "three", "four", "", "five", "six", ""],
            ["one", "two", "", "three", "four", "", "five", "six", ""],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six"],
            ["one", "two", "", "three", "four", "", "five", "six"],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six", ""],
            ["one", "Two", "", "threE", "four", "", "five", "siX", ""],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six"],
            ["one", "Two", "", "threE", "four", "", "five", "siX"],
        ),
        (
            ["one", "two", "", "thre", "four", "", "five", "six", ""],
            ["one", "two", "", "three", "four", "", "five", "six", ""],
        ),
        (
            ["one", "two", "", "thre", "four", "", "five", "six"],
            ["one", "two", "", "three", "four", "", "five", "six"],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six", ""],
            ["one", "two", ":)", "three", "four", "", "five", "six", ""],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six"],
            ["one", "two", ":)", "three", "four", "", "five", "six"],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six", ""],
            ["", "two", "", "three", "four", "", "five", "six", ""],
        ),
        (
            ["one", "two", "", "three", "four", "", "five", "six"],
            ["", "two", "", "three", "four", "", "five", "six"],
        ),
    ]

    for left, right in trials:
        assert ak.to_list(
            ak.Array(left, check_valid=True) == ak.Array(right, check_valid=True)
        ) == [x == y for x, y in zip(left, right)]


def test_string_equal2():
    assert ak.to_list(
        ak.Array(["one", "two", "three", "two", "two", "one"], check_valid=True)
        == "two"
    ) == [False, True, False, True, True, False]
