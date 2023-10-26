# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

numba = pytest.importorskip("numba")


def test_ArrayBuilder_inNumba():
    SOME_ATTRS = {"FOO": "BAR"}
    builder = ak.ArrayBuilder(behavior=SOME_ATTRS)

    @numba.njit
    def func(array):
        return array

    assert builder.behavior is SOME_ATTRS
    # In Python, when we create a dictionary literal like {'FOO': 'BAR'}, it
    # creates a new dictionary object. If we serialize this dictionary to
    # a JSON string, or to a tuple and then deserialize it, we get a new dictionary
    # object that is structurally identical to the original one, but it is not
    # the same object in terms of identity.

    # To check if two dictionaries are equal in terms of their contents,
    # we should use the == operator instead of is.
    assert func(builder).behavior == SOME_ATTRS
