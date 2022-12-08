# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import numpy as np
import pytest

import awkward as ak


def test_from_iter():
    assert ak.operations.from_iter([1 + 1j, 2 + 2j, 3 + 3j]).to_list() == [
        1 + 1j,
        2 + 2j,
        3 + 3j,
    ]
    assert ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]).to_list() == [
        [1 + 1j, 2 + 2j],
        [],
        [3 + 3j],
    ]

    # First encounter of a complex number should promote previous integers and
    # reals into complex numbers:
    assert ak.operations.from_iter([1, 2.2, 3 + 3j]).to_list() == [
        1.0 + 0j,
        2.2 + 0j,
        3.0 + 3j,
    ]
    assert ak.operations.from_iter([1, 3 + 3j]).to_list() == [
        1.0 + 0j,
        3.0 + 3j,
    ]

    # Just as the first encounter of a real number promotes previous integers
    # into reals:
    assert str(ak.operations.from_iter([1, 2.2]).type) == "2 * float64"
    assert ak.operations.from_iter([1, 2.2]).to_list() == [1.0, 2.2]
    builder = ak.highlevel.ArrayBuilder()
    assert str(builder.type) == "0 * unknown"
    builder.integer(1)
    assert str(builder.type) == "1 * int64"
    builder.real(2.2)
    assert str(builder.type) == "2 * float64"

    # For that matter, ArrayBuilder is missing a high-level interface to complex:
    builder.complex(3 + 3j)
    assert str(builder.type) == "3 * complex128"


def test_from_json():
    array = ak.operations.from_json('[{"r": 1.1, "i": 1.0}, {"r": 2.2, "i": 2.0}]')
    assert array.to_list() == [
        {"r": 1.1, "i": 1.0},
        {"r": 2.2, "i": 2.0},
    ]
    array = ak.operations.from_json(
        '[{"r": 1.1, "i": 1.0}, {"r": 2.2, "i": 2.0}]', complex_record_fields=("r", "i")
    )
    assert array.to_list() == [(1.1 + 1j), (2.2 + 2j)]

    # Somewhere in from_json, a handler that turns integer record fields into
    # parts of a complex number is missing.
    array = ak.operations.from_json(
        '[{"r": 1, "i": 1}, {"r": 2, "i": 2}]', complex_record_fields=("r", "i")
    )
    assert array.to_list() == [(1 + 1j), (2 + 2j)]

    # This should fail with some message like "complex number fields must be numbers,"
    # not "called 'end_record' without 'begin_record' at the same level before it."
    with pytest.raises(ValueError):
        array = ak.operations.from_json(
            '[{"r": [], "i": 1}, {"r": [1, 2], "i": 2}]',
            complex_record_fields=("r", "i"),
        )

    # These shouldn't be recognized as complex number records because they have
    # only one of the two fields.
    assert ak.operations.from_json(
        '[{"r": 1}, {"r": 2}]', complex_record_fields=("r", "i")
    ).to_list() == [{"r": 1}, {"r": 2}]
    assert ak.operations.from_json(
        '[{"i": 1}, {"i": 2}]', complex_record_fields=("r", "i")
    ).to_list() == [{"i": 1}, {"i": 2}]
    assert ak.operations.from_json(
        '[{"r": 1.1}, {"r": 2.2}]', complex_record_fields=("r", "i")
    ).to_list() == [{"r": 1.1}, {"r": 2.2}]
    assert ak.operations.from_json(
        '[{"i": 1.1}, {"i": 2.2}]', complex_record_fields=("r", "i")
    ).to_list() == [{"i": 1.1}, {"i": 2.2}]

    # In this one, the extra field should simply be ignored. A record with *at least*
    # the two specified fields should be recognized as a complex number, so that
    # the protocol can include a type marker, as some protocols do.
    array = ak.operations.from_json(
        '[{"r": 1.1, "i": 1.0, "another": []}, {"r": 2.2, "i": 2.0, "another": [1, 2, 3]}]',
        complex_record_fields=("r", "i"),
    )
    assert array.to_list() == [(1.1 + 1j), (2.2 + 2j)]


def test_to_json():
    # Complex numbers can't be converted to JSON without setting 'complex_record_fields',
    # but the error messages should refer to that name now. (I changed the name at
    # high-level, but not in the error messages emitted by C++ code.)
    with pytest.raises(TypeError) as err:
        ak.operations.to_json(ak.operations.from_iter([1 + 1j, 2 + 2j, 3 + 3j]))
    assert "not JSON serializable" in str(err)

    expectation = [{"r": 1.0, "i": 1.0}, {"r": 2.0, "i": 2.0}, {"r": 3.0, "i": 3.0}]
    assert expectation == json.loads(
        ak.operations.to_json(
            ak.operations.from_iter([1 + 1j, 2 + 2j, 3 + 3j]),
            complex_record_fields=("r", "i"),
        )
    )
    expectation = [
        [{"r": 1.0, "i": 1.0}, {"r": 2.0, "i": 2.0}],
        [],
        [{"r": 3.0, "i": 3.0}],
    ]
    assert expectation == json.loads(
        ak.operations.to_json(
            ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]),
            complex_record_fields=("r", "i"),
        )
    )


def test_reducers():
    assert (
        ak.operations.sum(ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]))
        == 6 + 6j
    )
    assert (
        ak.operations.prod(ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]))
        == -12 + 12j
    )

    assert ak.operations.sum(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        3 + 3j,
        0 + 0j,
        3 + 3j,
    ]
    assert ak.operations.prod(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        0 + 4j,
        1 + 0j,
        3 + 3j,
    ]

    assert ak.operations.count(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [2, 0, 1]
    assert ak.operations.count_nonzero(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [2, 0, 1]
    assert ak.operations.any(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        True,
        False,
        True,
    ]
    assert ak.operations.all(
        ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        True,
        True,
        True,
    ]
    assert ak.operations.any(
        ak.operations.from_iter([[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]]),
        axis=1,
    ).to_list() == [True, False, True]
    assert ak.operations.all(
        ak.operations.from_iter([[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]]),
        axis=1,
    ).to_list() == [False, True, True]


def test_minmax():
    assert (
        ak.operations.min(ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]))
        == 1 + 5j
    )
    assert (
        ak.operations.max(ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]))
        == 3 + 3j
    )

    assert ak.operations.min(
        ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        1 + 5j,
        None,
        3 + 3j,
    ]
    assert ak.operations.max(
        ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).to_list() == [
        2 + 4j,
        None,
        3 + 3j,
    ]

    assert ak.operations.argmin(
        ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).to_list() == [0, None, 0]
    assert ak.operations.argmax(
        ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).to_list() == [1, None, 0]


@pytest.mark.skip(reason="Remember to implement sorting for complex numbers.")
def test_sort():
    assert ak.operations.sort(
        ak.operations.from_iter([[2 + 4j, 1 + 5j], [], [3 + 3j]])
    ).to_list() == [
        [1 + 5j, 2 + 4j],
        [],
        [3 + 3j],
    ]
    assert ak.operations.argsort(
        ak.operations.from_iter([[2 + 4j, 1 + 5j], [], [3 + 3j]])
    ).to_list() == [
        [1, 0],
        [],
        [0],
    ]


def test_numpy():
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.operations.from_iter([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        ),
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
    )
    assert (
        str(
            ak.operations.to_numpy(
                ak.operations.from_iter([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
            ).dtype
        )
        == "complex128"
    )
    assert ak.highlevel.Array(
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    ).to_list() == [
        [(1 + 1j), (2 + 2j)],
        [(3 + 3j), (4 + 4j)],
    ]
