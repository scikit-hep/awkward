# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np

one = ak._v2.Array([[0.0, 1.1, 2.2, None], [], [3.3, 4.4]])
two = ak._v2.Array([[100, 200, 300, 400], [300], [400, 500]])


def test_behavior_forwarding_structure():
    three = ak._v2.operations.convert.from_iter(
        [[0.99999], [6], [2.99999]], highlevel=True
    )
    four = ak._v2.operations.convert.from_iter(
        [[1.00001], [6], [3.00001]], highlevel=True
    )
    mask1 = ak._v2.highlevel.Array([[True, True, False, False], [True], [True, False]])
    five = ak._v2.Array(["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"])

    six = ak._v2.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 3}, {"x": 3}], check_valid=True
    )
    seven = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)

    assert (
        ak._v2.operations.structure.argcartesian([one, two], behavior={})[0].behavior
        == {}
    )
    assert (
        ak._v2.operations.structure.argcombinations(one, 2, behavior={})[0].behavior
        == {}
    )
    assert ak._v2.operations.structure.argsort(one, behavior={})[0].behavior == {}

    assert (
        ak._v2.operations.structure.broadcast_arrays(
            5, ak._v2.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4]], behavior={})
        )[0].behavior
        == {}
    )

    assert ak._v2.operations.structure.cartesian([one], behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.structure.combinations(one, 2, behavior={})[0].behavior == {}
    )
    assert (
        ak._v2.operations.structure.concatenate([one, two], behavior={})[0].behavior
        == {}
    )

    assert ak._v2.operations.structure.fill_none(one, 42, behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.flatten(one, behavior={}).behavior == {}
    assert ak._v2.operations.structure.from_regular(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.full_like(one, 6, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.is_none(one, behavior={}).behavior == {}
    assert ak._v2.operations.structure.isclose(three, four, behavior={}).behavior == {}

    assert ak._v2.operations.structure.local_index(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.mask(two, mask1, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.nan_to_num(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.num(one, behavior={}).behavior == {}

    assert ak._v2.operations.structure.ones_like(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.packed(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.pad_none(one, 6, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.ravel(one, behavior={}).behavior == {}
    assert ak._v2.operations.structure.run_lengths(one, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.sort(one, behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.structure.strings_astype(
            five, np.float64, behavior={}
        ).behavior
        == {}
    )

    assert ak._v2.operations.structure.to_regular(three, behavior={})[0].behavior == {}

    assert ak._v2.operations.structure.unflatten(five, 2, behavior={})[0].behavior == {}
    assert (
        ak._v2.operations.structure.unzip(ak._v2.Array([{"x": 1}], behavior={}))[
            0
        ].behavior
        == {}
    )

    assert (
        ak._v2.operations.structure.values_astype(one, "float32", behavior={})[
            0
        ].behavior
        == {}
    )

    assert (
        ak._v2.operations.structure.where(
            ak._v2.highlevel.Array(
                [[True, True], [True, False], [True, False]], behavior={}
            )
        )[0].behavior
        == {}
    )
    assert (
        ak._v2.operations.structure.with_field(six, seven, where="y", behavior={})[
            0
        ].behavior
        == {}
    )
    assert (
        ak._v2.operations.structure.with_name(six, "cloud", behavior={})[0].behavior
        == {}
    )
    assert (
        ak._v2.operations.structure.without_parameters(
            ak._v2.operations.structure.with_parameter(
                one, "__array__", "cloud", behavior={}
            )
        )[0].behavior
        == {}
    )

    assert ak._v2.operations.structure.zeros_like(one, behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.zip([five, seven], behavior={})[0].behavior == {}


def test_behavior_forwarding_convert():
    assert (
        ak._v2.operations.convert.from_json_schema(
            " [ 1 ,2 ,3.0, 4, 5]  \n  ",
            {"type": "array", "items": {"type": "integer"}},
            behavior={},
        ).behavior
        == {}
    )


def test_behaviour_singletons_firsts():
    assert ak._v2.operations.structure.firsts([one, two], behavior={})[0].behavior == {}
    assert ak._v2.operations.structure.singletons(one, behavior={})[0].behavior == {}
