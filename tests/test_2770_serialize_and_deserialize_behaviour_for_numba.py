# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")


@numba.njit
def func(array):
    return array


def test_ArrayBuilder_behavior():
    SOME_ATTRS = {"FOO": "BAR"}
    builder = ak.ArrayBuilder(behavior=SOME_ATTRS)

    assert builder.behavior is SOME_ATTRS
    assert func(builder).behavior == SOME_ATTRS


def test_ArrayBuilder_non_picklable_behavior():
    def make_add_xyr():
        def add_xyr(left, right):
            x = left.x + right.x
            y = left.y + right.y
            return ak.zip(
                {
                    "x": x,
                    "y": y,
                    "r": np.sqrt(x**2 + y**2),
                },
                with_name="xyr",
            )

        return add_xyr

    behavior = {(np.add, "xyr", "xyr"): make_add_xyr()}
    builder = ak.ArrayBuilder(behavior=behavior)
    behavior_out = func(builder).behavior

    # Compare the dictionaries themselves
    # Note: 'behavior_out' is not 'behavior'
    if behavior_out == behavior:
        print("behavior_out is behavior")
    else:
        print("behavior_out is not behavior")

    # Define ufuncs
    ufunc_add = np.add

    # Compare the identity of the ufunc within the dictionaries
    # Note: 'ufunc_behavior_out[0]' is 'ufunc_add'
    ufunc_behavior_out = next(iter(behavior_out))
    ufunc_behavior = next(iter(behavior))

    if ufunc_behavior_out[0] is ufunc_add:
        print("ufunc_behavior_out[0] is ufunc_add")
    else:
        print("ufunc_behavior_out[0] is not ufunc_add")

    if ufunc_behavior[0] is ufunc_add:
        print("ufunc_behavior[0] is ufunc_add")
    else:
        print("ufunc_behavior[0] is not ufunc_add")

    # Compare the unique identifiers of the lambda functions
    # Note: Lambda functions have different identities
    lambda_behavior_out = next(iter(behavior_out.values()))
    lambda_behavior = next(iter(behavior.values()))

    if id(lambda_behavior_out) == id(lambda_behavior):
        print("Lambda functions have the same identity")
    else:
        print("Lambda functions have different identities")

    @numba.njit
    def make_ab(builder):
        builder.begin_record("xyz")
        builder.field("x").integer(3)
        builder.field("y").integer(4)
        builder.field("z").integer(3)
        builder.end_record()

        builder.begin_record("xyz")
        builder.field("x").integer(3)
        builder.field("y").integer(4)
        builder.field("z").integer(3)
        builder.end_record()
        return builder

    result = make_ab(builder).snapshot()

    print(result)
    assert result.behavior == make_ab(builder).behavior
