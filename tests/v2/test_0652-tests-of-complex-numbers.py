# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_from_iter():
    assert ak._v2.operations.convert.from_iter([1 + 1j, 2 + 2j, 3 + 3j]).tolist() == [
        1 + 1j,
        2 + 2j,
        3 + 3j,
    ]
    assert ak._v2.operations.convert.from_iter(
        [[1 + 1j, 2 + 2j], [], [3 + 3j]]
    ).tolist() == [
        [1 + 1j, 2 + 2j],
        [],
        [3 + 3j],
    ]

    # First encounter of a complex number should promote previous integers and
    # reals into complex numbers:
    assert ak._v2.operations.convert.from_iter([1, 2.2, 3 + 3j]).tolist() == [
        1.0 + 0j,
        2.2 + 0j,
        3.0 + 3j,
    ]
    assert ak._v2.operations.convert.from_iter([1, 3 + 3j]).tolist() == [
        1.0 + 0j,
        3.0 + 3j,
    ]

    # Just as the first encounter of a real number promotes previous integers
    # into reals:
    assert str(ak._v2.operations.convert.from_iter([1, 2.2]).type) == "2 * float64"
    assert ak._v2.operations.convert.from_iter([1, 2.2]).tolist() == [1.0, 2.2]
    builder = ak._v2.highlevel.ArrayBuilder()
    assert str(builder.type) == "0 * unknown"
    builder.integer(1)
    assert str(builder.type) == "1 * int64"
    builder.real(2.2)
    assert str(builder.type) == "2 * float64"

    # For that matter, ArrayBuilder is missing a high-level interface to complex:
    builder.complex(3 + 3j)
    assert str(builder.type) == "3 * complex128"


@pytest.mark.skip(
    reason="awkward/_v2/operations/convert/ak_from_json.py:26: NotImplementedError"
)
def test_from_json():
    array = ak._v2.operations.convert.from_json(
        '[{"r": 1.1, "i": 1.0}, {"r": 2.2, "i": 2.0}]'
    )
    assert array.tolist() == [
        {"r": 1.1, "i": 1.0},
        {"r": 2.2, "i": 2.0},
    ]
    array = ak._v2.operations.convert.from_json(
        '[{"r": 1.1, "i": 1.0}, {"r": 2.2, "i": 2.0}]', complex_record_fields=("r", "i")
    )
    assert array.tolist() == [(1.1 + 1j), (2.2 + 2j)]

    # Somewhere in from_json, a handler that turns integer record fields into
    # parts of a complex number is missing.
    array = ak._v2.operations.convert.from_json(
        '[{"r": 1, "i": 1}, {"r": 2, "i": 2}]', complex_record_fields=("r", "i")
    )
    assert array.tolist() == [(1 + 1j), (2 + 2j)]

    # This should fail with some message like "complex number fields must be numbers,"
    # not "called 'end_record' without 'begin_record' at the same level before it."
    with pytest.raises(ValueError) as err:
        array = ak._v2.operations.convert.from_json(
            '[{"r": [], "i": 1}, {"r": [1, 2], "i": 2}]',
            complex_record_fields=("r", "i"),
        )
        assert array["r"].type == array["i"].type
    assert "Complex number fields must be numbers" in str(err)

    # These shouldn't be recognized as complex number records because they have
    # only one of the two fields.
    assert ak._v2.operations.convert.from_json(
        '[{"r": 1}, {"r": 2}]', complex_record_fields=("r", "i")
    ).tolist() == [{"r": 1}, {"r": 2}]
    assert ak._v2.operations.convert.from_json(
        '[{"i": 1}, {"i": 2}]', complex_record_fields=("r", "i")
    ).tolist() == [{"i": 1}, {"i": 2}]
    assert ak._v2.operations.convert.from_json(
        '[{"r": 1.1}, {"r": 2.2}]', complex_record_fields=("r", "i")
    ).tolist() == [{"r": 1.1}, {"r": 2.2}]
    assert ak._v2.operations.convert.from_json(
        '[{"i": 1.1}, {"i": 2.2}]', complex_record_fields=("r", "i")
    ).tolist() == [{"i": 1.1}, {"i": 2.2}]

    # In this one, the extra field should simply be ignored. A record with *at least*
    # the two specified fields should be recognized as a complex number, so that
    # the protocol can include a type marker, as some protocols do.
    array = ak._v2.operations.convert.from_json(
        '[{"r": 1.1, "i": 1.0, "another": []}, {"r": 2.2, "i": 2.0, "another": [1, 2, 3]}]',
        complex_record_fields=("r", "i"),
    )
    assert array.tolist() == [(1.1 + 1j), (2.2 + 2j)]


@pytest.mark.skip(
    reason="awkward/_v2/operations/convert/ak_to_json.py:21: NotImplementedError"
)
def test_to_json():
    # Complex numbers can't be converted to JSON without setting 'complex_record_fields',
    # but the error messages should refer to that name now. (I changed the name at
    # high-level, but not in the error messages emitted by C++ code.)
    with pytest.raises(ValueError) as err:
        ak._v2.operations.convert.to_json(
            ak._v2.operations.convert.from_iter([1 + 1j, 2 + 2j, 3 + 3j])
        )
    assert "needs both" not in str(err)

    expectation = [{"r": 1.0, "i": 1.0}, {"r": 2.0, "i": 2.0}, {"r": 3.0, "i": 3.0}]
    assert expectation == json.loads(
        ak._v2.operations.convert.to_json(
            ak._v2.operations.convert.from_iter([1 + 1j, 2 + 2j, 3 + 3j]),
            complex_record_fields=("r", "i"),
        )
    )
    expectation = [
        [{"r": 1.0, "i": 1.0}, {"r": 2.0, "i": 2.0}],
        [],
        [{"r": 3.0, "i": 3.0}],
    ]
    assert expectation == json.loads(
        ak._v2.operations.convert.to_json(
            ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]),
            complex_record_fields=("r", "i"),
        )
    )


def test_reducers():
    assert (
        ak._v2.operations.reducers.sum(
            ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]])
        )
        == 6 + 6j
    )
    assert (
        ak._v2.operations.reducers.prod(
            ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]])
        )
        == -12 + 12j
    )

    assert ak._v2.operations.reducers.sum(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        3 + 3j,
        0 + 0j,
        3 + 3j,
    ]
    assert ak._v2.operations.reducers.prod(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        0 + 4j,
        1 + 0j,
        3 + 3j,
    ]

    assert ak._v2.operations.reducers.count(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [2, 0, 1]
    assert ak._v2.operations.reducers.count_nonzero(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [2, 0, 1]
    assert ak.any(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        True,
        False,
        True,
    ]
    assert ak._v2.operations.reducers.all(
        ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        True,
        True,
        True,
    ]
    assert (
        ak._v2.operations.reducers.any(
            ak._v2.operations.convert.from_iter(
                [[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]]
            ),
            axis=1,
        ).tolist()
        == [True, False, True]
    )
    assert (
        ak._v2.operations.reducers.all(
            ak._v2.operations.convert.from_iter(
                [[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]]
            ),
            axis=1,
        ).tolist()
        == [False, True, True]
    )


def test_minmax():
    assert (
        ak._v2.operations.reducers.min(
            ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]])
        )
        == 1 + 5j
    )
    assert (
        ak._v2.operations.reducers.max(
            ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]])
        )
        == 3 + 3j
    )

    assert ak._v2.operations.reducers.min(
        ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        1 + 5j,
        None,
        3 + 3j,
    ]
    assert ak._v2.operations.reducers.max(
        ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).tolist() == [
        2 + 4j,
        None,
        3 + 3j,
    ]

    assert ak._v2.operations.reducers.argmin(
        ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).tolist() == [0, None, 0]
    assert ak._v2.operations.reducers.argmax(
        ak._v2.operations.convert.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]]), axis=1
    ).tolist() == [1, None, 0]


@pytest.mark.skip(
    reason="awkward/_v2/operations/structure/ak_sort.py:12: NotImplementedError"
)
def test_sort():
    assert ak._v2.operations.structure.sort(
        ak._v2.operations.convert.from_iter([[2 + 4j, 1 + 5j], [], [3 + 3j]])
    ).tolist() == [
        [1 + 5j, 2 + 4j],
        [],
        [3 + 3j],
    ]
    assert ak._v2.operations.structure.argsort(
        ak._v2.operations.convert.from_iter([[2 + 4j, 1 + 5j], [], [3 + 3j]])
    ).tolist() == [
        [1, 0],
        [],
        [0],
    ]


def test_numpy():
    assert np.array_equal(
        ak._v2.operations.convert.to_numpy(
            ak._v2.operations.convert.from_iter([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        ),
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
    )
    assert (
        str(
            ak._v2.operations.convert.to_numpy(
                ak._v2.operations.convert.from_iter(
                    [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]
                )
            ).dtype
        )
        == "complex128"
    )
    assert ak._v2.highlevel.Array(
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    ).tolist() == [
        [(1 + 1j), (2 + 2j)],
        [(3 + 3j), (4 + 4j)],
    ]
