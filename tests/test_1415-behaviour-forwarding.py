# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np

one = ak._v2.Array([[0.0, 1.1, 2.2, None], [], [3.3, 4.4]])
two = ak._v2.Array([[100, 200, 300, 400], [300], [400, 500]])


def test_behavior_forwarding_structure():
    three = ak._v2.operations.from_iter([[0.99999], [6], [2.99999]], highlevel=True)
    four = ak._v2.operations.from_iter([[1.00001], [6], [3.00001]], highlevel=True)
    mask1 = ak._v2.highlevel.Array([[True, True, False, False], [True], [True, False]])
    five = ak._v2.Array(["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"])

    six = ak._v2.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 3}, {"x": 3}], check_valid=True
    )
    seven = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)

    assert ak._v2.operations.argcartesian([one, two], behavior={})[0].behavior == {}
    assert ak._v2.operations.argcombinations(one, 2, behavior={})[0].behavior == {}
    assert ak._v2.operations.argsort(one, behavior={})[0].behavior == {}

    assert (
        ak._v2.operations.broadcast_arrays(
            5, ak._v2.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]], behavior={})
        )[0].behavior
        == {}
    )

    assert ak._v2.operations.cartesian([one], behavior={})[0].behavior == {}
    assert ak._v2.operations.combinations(one, 2, behavior={})[0].behavior == {}
    assert ak._v2.operations.concatenate([one, two], behavior={})[0].behavior == {}

    assert ak._v2.operations.fill_none(one, 42, behavior={})[0].behavior == {}
    assert ak._v2.operations.flatten(one, behavior={}).behavior == {}
    assert ak._v2.operations.from_regular(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.full_like(one, 6, behavior={})[0].behavior == {}

    assert ak._v2.operations.is_none(one, behavior={}).behavior == {}
    assert ak._v2.operations.isclose(three, four, behavior={}).behavior == {}

    assert ak._v2.operations.local_index(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.mask(two, mask1, behavior={})[0].behavior == {}

    assert ak._v2.operations.nan_to_num(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.num(one, behavior={}).behavior == {}

    assert ak._v2.operations.ones_like(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.packed(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.pad_none(one, 6, behavior={})[0].behavior == {}

    assert ak._v2.operations.ravel(one, behavior={}).behavior == {}
    assert ak._v2.operations.run_lengths(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.sort(one, behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.strings_astype(five, np.float64, behavior={}).behavior == {}
    )

    assert ak._v2.operations.to_regular(three, behavior={})[0].behavior == {}

    assert ak._v2.operations.unflatten(five, 2, behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.unzip(ak._v2.Array([{"x": 1}], behavior={}))[0].behavior == {}
    )

    assert (
        ak._v2.operations.values_astype(one, "float32", behavior={})[0].behavior == {}
    )

    assert (
        ak._v2.operations.where(
            ak._v2.highlevel.Array(
                [[True, True], [True, False], [True, False]], behavior={}
            )
        )[0].behavior
        == {}
    )
    assert (
        ak._v2.operations.with_field(six, seven, where="y", behavior={})[0].behavior
        == {}
    )
    assert ak._v2.operations.with_name(six, "cloud", behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.without_parameters(
            ak._v2.operations.with_parameter(one, "__array__", "cloud", behavior={})
        )[0].behavior
        == {}
    )

    assert ak._v2.operations.zeros_like(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.zip([five, seven], behavior={})[0].behavior == {}


def test_behavior_forwarding_convert():
    assert (
        ak._v2.operations.from_json(
            " [ 1 ,2 ,3.0, 4, 5]  \n  ",
            schema={"type": "array", "items": {"type": "number"}},
            behavior={},
        ).behavior
        == {}
    )


def test_behaviour_singletons_firsts():
    assert ak._v2.operations.firsts([one, two], behavior={})[0].behavior == {}
    assert ak._v2.operations.singletons(one, behavior={})[0].behavior == {}
