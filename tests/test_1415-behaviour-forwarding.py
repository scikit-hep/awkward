# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("operation_behavior", [None, {"other-type": ak.Record}])
def test_behavior_forwarding_structure(operation_behavior):
    """Ensure that explicit behaviors win, or the existing behavior is maintained"""
    local_behavior = {"some-type": ak.Array}
    merged_behavior = (
        local_behavior if operation_behavior is None else operation_behavior
    )
    one = ak.Array([[0.0, 1.1, 2.2, None], [], [3.3, 4.4]], behavior=local_behavior)
    two = ak.Array([[100, 200, 300, 400], [300], [400, 500]], behavior=local_behavior)
    three = ak.operations.from_iter(
        [[0.99999], [6], [2.99999]], highlevel=True, behavior=local_behavior
    )
    four = ak.operations.from_iter(
        [[1.00001], [6], [3.00001]], highlevel=True, behavior=local_behavior
    )
    mask1 = ak.highlevel.Array(
        [[True, True, False, False], [True], [True, False]], behavior=local_behavior
    )
    mask2 = ak.highlevel.Array(
        [[True, True], [True, False], [True, False]], behavior=local_behavior
    )
    five = ak.Array(
        ["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"], behavior=local_behavior
    )

    six = ak.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 3}, {"x": 3}],
        check_valid=True,
        behavior=local_behavior,
    )
    seven = ak.Array(
        [1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True, behavior=local_behavior
    )

    assert (
        ak.operations.argcartesian([one, two], behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.argcombinations(one, 2, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.argsort(one, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )

    assert (
        ak.operations.broadcast_arrays(5, one, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )

    assert (
        ak.operations.cartesian([one], behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.combinations(one, 2, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.concatenate([one, two], behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.firsts(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.fill_none(one, 42, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.flatten(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.from_regular(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.from_json(
            " [ 1 ,2 ,3.0, 4, 5]  \n  ",
            schema={"type": "array", "items": {"type": "number"}},
            behavior=local_behavior,
        ).behavior
        == local_behavior
    )
    assert (
        ak.operations.full_like(one, 6, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.is_none(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.isclose(three, four, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.local_index(one, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.mask(two, mask1, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.nan_to_num(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.num(one, behavior=operation_behavior).behavior == merged_behavior
    )

    assert (
        ak.operations.ones_like(one, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.to_packed(one, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )
    assert (
        ak.operations.pad_none(one, 6, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.ravel(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.run_lengths(one, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.singletons(one, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.sort(one, behavior=operation_behavior).behavior == merged_behavior
    )
    assert (
        ak.operations.strings_astype(
            five, np.float64, behavior=operation_behavior
        ).behavior
        == merged_behavior
    )

    def transformer(layout, **kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(layout.data * 2)

    assert (
        ak.operations.transform(
            transformer, three, behavior=operation_behavior
        ).behavior
        == merged_behavior
    )

    assert (
        ak.operations.to_regular(three, behavior=operation_behavior).behavior
        == merged_behavior
    )

    assert (
        ak.operations.unflatten(five, 2, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.unzip(six, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )

    assert (
        ak.operations.values_astype(
            one, "float32", behavior=operation_behavior
        ).behavior
        == merged_behavior
    )

    # Different `where` implementations
    assert (
        ak.operations.where(mask2, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )
    assert (
        ak.operations.where(mask1, ~mask1, mask1, behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.with_field(
            six, seven, where="y", behavior=operation_behavior
        ).behavior
        == merged_behavior
    )
    assert (
        ak.operations.with_name(six, "cloud", behavior=operation_behavior).behavior
        == merged_behavior
    )
    assert (
        ak.operations.without_parameters(
            ak.operations.with_parameter(
                one, "__array__", "cloud", behavior=operation_behavior
            )
        ).behavior
        == merged_behavior
    )

    assert (
        ak.operations.zeros_like(one, behavior=operation_behavior)[0].behavior
        == merged_behavior
    )
    assert (
        ak.operations.zip([five, seven], behavior=operation_behavior)[0].behavior
        == merged_behavior
    )
