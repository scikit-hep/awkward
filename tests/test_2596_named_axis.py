# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._namedaxis import _get_named_axis


def test_constructor():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis=("x", "y"))
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 0, "y": 1}
    assert array.positional_axis == (0, 1)

    array = ak.Array([[1, 2], [3], [], [4, 5, 6]], named_axis={"x": 0, "y": 1})
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 0, "y": 1}
    assert array.positional_axis == (0, 1)


def test_with_named_axis():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    assert not _get_named_axis(array)
    assert array.named_axis == {}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis=("x", "y"))
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 0, "y": 1}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis=("x", None))
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 0}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis=(None, "x"))
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 1}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"x": 0, "y": 1})
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 0, "y": 1}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"x": 1})
    assert _get_named_axis(array)
    assert array.named_axis == {"x": 1}
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"y": -1})
    assert _get_named_axis(array)
    assert array.named_axis == {"y": -1}
    assert array.positional_axis == (0, 1)

    # This is possible in a future version of named axis, but currently only strings are supported
    # from dataclasses import dataclass

    # @dataclass(frozen=True)
    # class exotic_axis:
    #     attr: str

    # ax1 = exotic_axis(attr="I'm not the type of axis that you're used to")
    # ax2 = exotic_axis(attr="...me neither!")

    # array = ak.with_named_axis(array, named_axis=(ax1, ax2))
    # assert array.named_axis == (ax1, ax2)
    # assert array.positional_axis == (0, 1)


def test_named_axis_indexing():
    array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y", "z"))

    # test indexing
    assert ak.all(array[...] == named_array[...])
    assert ak.all(array[()] == named_array[()])

    assert ak.all(array[None, :, :, :] == named_array[None, :, :, :])
    assert ak.all(array[:, None, :, :] == named_array[:, None, :, :])
    assert ak.all(array[:, :, None, :] == named_array[:, :, None, :])
    assert ak.all(array[:, :, :, None] == named_array[:, :, :, None])

    assert ak.all(array[0, :, :] == named_array[{"x": 0}])
    assert ak.all(array[:, 0, :] == named_array[{"y": 0}])
    assert ak.all(array[:, :, 0] == named_array[{"z": 0}])

    assert ak.all(array[0, :, :] == named_array[{0: 0}])
    assert ak.all(array[:, 0, :] == named_array[{1: 0}])
    assert ak.all(array[:, :, 0] == named_array[{2: 0}])

    assert ak.all(array[0, :, :] == named_array[{-3: 0}])
    assert ak.all(array[:, 0, :] == named_array[{-2: 0}])
    assert ak.all(array[:, :, 0] == named_array[{-1: 0}])

    assert ak.all(array[0, 0, :] == named_array[{"x": 0, "y": 0}])
    assert ak.all(array[0, :, 0] == named_array[{"x": 0, "z": 0}])
    assert ak.all(array[:, 0, 0] == named_array[{"y": 0, "z": 0}])
    assert array[0, 0, 0] == named_array[{"x": 0, "y": 0, "z": 0}]

    assert ak.all(array[slice(0, 1), :, :] == named_array[{"x": slice(0, 1)}])
    assert ak.all(array[:, slice(0, 1), :] == named_array[{"y": slice(0, 1)}])
    assert ak.all(array[:, :, slice(0, 1)] == named_array[{"z": slice(0, 1)}])

    assert ak.all(array[0, :, slice(0, 1)] == named_array[{"x": 0, "z": slice(0, 1)}])
    assert ak.all(array[:, 0, slice(0, 1)] == named_array[{"y": 0, "z": slice(0, 1)}])
    assert ak.all(array[slice(0, 1), 0, :] == named_array[{"x": slice(0, 1), "y": 0}])

    assert ak.all(array[array > 3] == named_array[named_array > 3])

    # test naming propagation
    assert (
        named_array[...].named_axis
        == named_array.named_axis
        == {"x": 0, "y": 1, "z": 2}
    )
    assert (
        named_array[()].named_axis == named_array.named_axis == {"x": 0, "y": 1, "z": 2}
    )

    # single int as index slices always only first dim
    assert named_array[0].named_axis == {"y": 0, "z": 1}
    assert named_array[1].named_axis == {"y": 0, "z": 1}
    assert named_array[2].named_axis == {"y": 0, "z": 1}
    assert named_array[3].named_axis == {"y": 0, "z": 1}

    assert named_array[None, :, :, :].named_axis == {"x": 1, "y": 2, "z": 3}
    assert named_array[:, None, :, :].named_axis == {"x": 0, "y": 2, "z": 3}
    assert named_array[:, :, None, :].named_axis == {"x": 0, "y": 1, "z": 3}
    assert named_array[:, :, :, None].named_axis == {"x": 0, "y": 1, "z": 2}

    assert named_array[None, ...].named_axis == {"x": 1, "y": 2, "z": 3}
    assert named_array[:, None, ...].named_axis == {"x": 0, "y": 2, "z": 3}
    assert named_array[..., None, :].named_axis == {"x": 0, "y": 1, "z": 3}
    assert named_array[..., None].named_axis == {"x": 0, "y": 1, "z": 2}

    assert (
        named_array[0, :, :].named_axis
        == named_array[{"x": 0}].named_axis
        == {"y": 0, "z": 1}
    )
    assert (
        named_array[:, 0, :].named_axis
        == named_array[{"y": 0}].named_axis
        == {"x": 0, "z": 1}
    )
    assert (
        named_array[:, :, 0].named_axis
        == named_array[{"z": 0}].named_axis
        == {"x": 0, "y": 1}
    )

    assert (
        named_array[0, ...].named_axis
        == named_array[{"x": 0}].named_axis
        == {"y": 0, "z": 1}
    )
    assert (
        named_array[:, 0, :].named_axis
        == named_array[{"y": 0}].named_axis
        == {"x": 0, "z": 1}
    )
    assert (
        named_array[..., 0].named_axis
        == named_array[{"z": 0}].named_axis
        == {"x": 0, "y": 1}
    )

    assert named_array[{0: 0}].named_axis == {"y": 0, "z": 1}
    assert named_array[{1: 0}].named_axis == {"x": 0, "z": 1}
    assert named_array[{2: 0}].named_axis == {"x": 0, "y": 1}

    assert named_array[{-3: 0}].named_axis == {"y": 0, "z": 1}
    assert named_array[{-2: 0}].named_axis == {"x": 0, "z": 1}
    assert named_array[{-1: 0}].named_axis == {"x": 0, "y": 1}

    assert (
        named_array[0, 0, :].named_axis
        == named_array[{"x": 0, "y": 0}].named_axis
        == {"z": 0}
    )
    assert (
        named_array[0, :, 0].named_axis
        == named_array[{"x": 0, "z": 0}].named_axis
        == {"y": 0}
    )
    assert (
        named_array[:, 0, 0].named_axis
        == named_array[{"y": 0, "z": 0}].named_axis
        == {"x": 0}
    )
    assert not _get_named_axis(named_array[0, 0, 0])
    assert not _get_named_axis(named_array[{"x": 0, "y": 0, "z": 0}])

    assert (
        named_array[slice(0, 1), :, :].named_axis
        == named_array[{"x": slice(0, 1)}].named_axis
        == {"x": 0, "y": 1, "z": 2}
    )
    assert (
        named_array[:, slice(0, 1), :].named_axis
        == named_array[{"y": slice(0, 1)}].named_axis
        == {"x": 0, "y": 1, "z": 2}
    )
    assert (
        named_array[:, :, slice(0, 1)].named_axis
        == named_array[{"z": slice(0, 1)}].named_axis
        == {"x": 0, "y": 1, "z": 2}
    )

    assert (
        named_array[0, :, slice(0, 1)].named_axis
        == named_array[{"x": 0, "z": slice(0, 1)}].named_axis
        == {"y": 0, "z": 1}
    )
    assert (
        named_array[:, 0, slice(0, 1)].named_axis
        == named_array[{"y": 0, "z": slice(0, 1)}].named_axis
        == {"x": 0, "z": 1}
    )
    assert (
        named_array[slice(0, 1), 0, :].named_axis
        == named_array[{"x": slice(0, 1), "y": 0}].named_axis
        == {"x": 0, "z": 1}
    )


def test_negative_named_axis_indexing():
    array = ak.Array([[[1, 2]], [[3]], [[4]], [[5, 6], [7]]])

    named_array = ak.with_named_axis(array, named_axis={"x": -3, "y": -2, "z": -1})

    # test indexing
    assert ak.all(array[...] == named_array[...])
    assert ak.all(array[()] == named_array[()])

    assert ak.all(array[None, :, :, :] == named_array[None, :, :, :])
    assert ak.all(array[:, None, :, :] == named_array[:, None, :, :])
    assert ak.all(array[:, :, None, :] == named_array[:, :, None, :])
    assert ak.all(array[:, :, :, None] == named_array[:, :, :, None])

    assert ak.all(array[0, :, :] == named_array[{"x": 0}])
    assert ak.all(array[:, 0, :] == named_array[{"y": 0}])
    assert ak.all(array[:, :, 0] == named_array[{"z": 0}])

    assert ak.all(array[0, :, :] == named_array[{0: 0}])
    assert ak.all(array[:, 0, :] == named_array[{1: 0}])
    assert ak.all(array[:, :, 0] == named_array[{2: 0}])

    assert ak.all(array[0, :, :] == named_array[{-3: 0}])
    assert ak.all(array[:, 0, :] == named_array[{-2: 0}])
    assert ak.all(array[:, :, 0] == named_array[{-1: 0}])

    assert ak.all(array[0, 0, :] == named_array[{"x": 0, "y": 0}])
    assert ak.all(array[0, :, 0] == named_array[{"x": 0, "z": 0}])
    assert ak.all(array[:, 0, 0] == named_array[{"y": 0, "z": 0}])
    assert array[0, 0, 0] == named_array[{"x": 0, "y": 0, "z": 0}]

    assert ak.all(array[slice(0, 1), :, :] == named_array[{"x": slice(0, 1)}])
    assert ak.all(array[:, slice(0, 1), :] == named_array[{"y": slice(0, 1)}])
    assert ak.all(array[:, :, slice(0, 1)] == named_array[{"z": slice(0, 1)}])

    assert ak.all(array[0, :, slice(0, 1)] == named_array[{"x": 0, "z": slice(0, 1)}])
    assert ak.all(array[:, 0, slice(0, 1)] == named_array[{"y": 0, "z": slice(0, 1)}])
    assert ak.all(array[slice(0, 1), 0, :] == named_array[{"x": slice(0, 1), "y": 0}])

    assert ak.all(array[array > 3] == named_array[named_array > 3])

    # test naming propagation
    assert (
        named_array[...].named_axis
        == named_array.named_axis
        == {"x": -3, "y": -2, "z": -1}
    )
    assert (
        named_array[()].named_axis
        == named_array.named_axis
        == {"x": -3, "y": -2, "z": -1}
    )

    assert named_array[None, :, :, :].named_axis == {"x": -3, "y": -2, "z": -1}
    assert named_array[:, None, :, :].named_axis == {"x": -4, "y": -2, "z": -1}
    assert named_array[:, :, None, :].named_axis == {"x": -4, "y": -3, "z": -1}
    assert named_array[:, :, :, None].named_axis == {"x": -4, "y": -3, "z": -2}

    assert named_array[None, ...].named_axis == {"x": -3, "y": -2, "z": -1}
    assert named_array[:, None, ...].named_axis == {"x": -4, "y": -2, "z": -1}
    assert named_array[..., None, :].named_axis == {"x": -4, "y": -3, "z": -1}
    assert named_array[..., None].named_axis == {"x": -4, "y": -3, "z": -2}

    assert (
        named_array[0, :, :].named_axis
        == named_array[{"x": 0}].named_axis
        == {"y": -2, "z": -1}
    )
    assert (
        named_array[:, 0, :].named_axis
        == named_array[{"y": 0}].named_axis
        == {"x": -2, "z": -1}
    )
    assert (
        named_array[:, :, 0].named_axis
        == named_array[{"z": 0}].named_axis
        == {"x": -2, "y": -1}
    )

    assert (
        named_array[0, ...].named_axis
        == named_array[{"x": 0}].named_axis
        == {"y": -2, "z": -1}
    )
    assert (
        named_array[..., 0].named_axis
        == named_array[{"z": 0}].named_axis
        == {"x": -2, "y": -1}
    )

    assert named_array[{0: 0}].named_axis == {"y": -2, "z": -1}
    assert named_array[{1: 0}].named_axis == {"x": -2, "z": -1}
    assert named_array[{2: 0}].named_axis == {"x": -2, "y": -1}

    assert named_array[{-3: 0}].named_axis == {"y": -2, "z": -1}
    assert named_array[{-2: 0}].named_axis == {"x": -2, "z": -1}
    assert named_array[{-1: 0}].named_axis == {"x": -2, "y": -1}

    assert (
        named_array[0, 0, :].named_axis
        == named_array[{"x": 0, "y": 0}].named_axis
        == {"z": -1}
    )
    assert (
        named_array[0, :, 0].named_axis
        == named_array[{"x": 0, "z": 0}].named_axis
        == {"y": -1}
    )
    assert (
        named_array[:, 0, 0].named_axis
        == named_array[{"y": 0, "z": 0}].named_axis
        == {"x": -1}
    )
    assert not _get_named_axis(named_array[0, 0, 0])
    assert not _get_named_axis(named_array[{"x": 0, "y": 0, "z": 0}])

    assert (
        named_array[slice(0, 1), :, :].named_axis
        == named_array[{"x": slice(0, 1)}].named_axis
        == {"x": -3, "y": -2, "z": -1}
    )
    assert (
        named_array[:, slice(0, 1), :].named_axis
        == named_array[{"y": slice(0, 1)}].named_axis
        == {"x": -3, "y": -2, "z": -1}
    )
    assert (
        named_array[:, :, slice(0, 1)].named_axis
        == named_array[{"z": slice(0, 1)}].named_axis
        == {"x": -3, "y": -2, "z": -1}
    )

    assert (
        named_array[0, :, slice(0, 1)].named_axis
        == named_array[{"x": 0, "z": slice(0, 1)}].named_axis
        == {"y": -2, "z": -1}
    )
    assert (
        named_array[:, 0, slice(0, 1)].named_axis
        == named_array[{"y": 0, "z": slice(0, 1)}].named_axis
        == {"x": -2, "z": -1}
    )
    assert (
        named_array[slice(0, 1), 0, :].named_axis
        == named_array[{"x": slice(0, 1), "y": 0}].named_axis
        == {"x": -2, "z": -1}
    )


def test_named_axis_right_broadcasting():
    # [NumPy-style] rightbroadcasting: (n, m) -> (1, n, m)
    a = ak.Array([1])  # (1,)
    b = ak.Array([[10, 20], [30, 40], [50, 60]])  # (3, 2)

    na = ak.with_named_axis(a, named_axis={"y": 0})
    nb = ak.with_named_axis(b, named_axis={"x": 0, "y": 1})

    naa, nbb = ak.broadcast_arrays(na, nb)

    assert naa.named_axis == nbb.named_axis == {"x": 0, "y": 1}

    na = ak.with_named_axis(a, named_axis={"y": -1})
    nb = ak.with_named_axis(b, named_axis={"y": -2, "x": -1})

    naa, nbb = ak.broadcast_arrays(na, nb)

    assert naa.named_axis == nbb.named_axis == {"y": -2, "x": -1}


def test_named_axis_left_broadcasting():
    # [Awkward-style] leftbroadcasting: (n, m) -> (n, m, 1)
    a = ak.Array([[[0, 1, 2], [], [3, 4]], [], [[5], [6, 7, 8, 9]]])  # (3, var, var)
    b = ak.Array([[10, 20, 30], [], [40, 50]])  # (3, var)

    na = ak.with_named_axis(a, named_axis=("x", "y", "z"))
    nb = ak.with_named_axis(b, named_axis=("x", "y"))

    naa, nbb = ak.broadcast_arrays(na, nb)

    assert naa.named_axis == nbb.named_axis == {"x": 0, "y": 1, "z": 2}

    na = ak.with_named_axis(a, named_axis={"x": -3, "y": -2, "z": -1})
    nb = ak.with_named_axis(b, named_axis={"x": -2, "y": -1})

    naa, nbb = ak.broadcast_arrays(na, nb)

    assert naa.named_axis == nbb.named_axis == {"x": -3, "y": -2, "z": -1}

    # this is not allowed!
    a = ak.with_named_axis(ak.Array([[1, 2], [3, 4]]), ("x", "y"))  # {"x": 0, "y": 1}
    asum = ak.sum(a, axis="x")  # {"y": 0}

    with pytest.raises(ValueError):
        _ = a + asum

    # this is allowed!
    a = ak.with_named_axis(ak.Array([[1, 2], [3, 4]]), ("x", "y"))  # {"x": 0, "y": 1}
    asum = ak.sum(a, axis="y")  # {"x": 0}

    assert (a + asum).named_axis == {"x": 0, "y": 1}


def test_named_axis_unary_ufuncs():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    assert (-named_array).named_axis == named_array.named_axis
    assert (+named_array).named_axis == named_array.named_axis
    assert (~named_array).named_axis == named_array.named_axis
    assert abs(named_array).named_axis == named_array.named_axis


def test_named_axis_binary_ufuncs():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = ak.with_named_axis(array, named_axis=(None, "y"))
    named_array2 = ak.with_named_axis(array, named_axis=("x", None))
    named_array3 = ak.with_named_axis(array, named_axis=("x", "y"))

    # just for addition, the rest is the same
    # __add__
    assert (array + array).named_axis == {}
    assert (named_array1 + array).named_axis == {"y": 1}
    assert (named_array2 + array).named_axis == {"x": 0}
    assert (named_array3 + array).named_axis == {"x": 0, "y": 1}

    assert (named_array1 + named_array2).named_axis == {"x": 0, "y": 1}
    assert (named_array3 + named_array3).named_axis == {"x": 0, "y": 1}

    # __radd__
    assert (array + named_array1).named_axis == {"y": 1}
    assert (array + named_array2).named_axis == {"x": 0}
    assert (array + named_array3).named_axis == {"x": 0, "y": 1}

    a = ak.with_named_axis(array, named_axis=("x", None))
    b = ak.with_named_axis(array, named_axis=("y", None))
    with pytest.raises(
        ValueError,
        match="The named axes are incompatible. Got: x and y for positional axis 0",
    ):
        _ = a + b

    a = ak.with_named_axis(array, named_axis=(None, "x"))
    b = ak.with_named_axis(array, named_axis=(None, "y"))
    with pytest.raises(
        ValueError,
        match="The named axes are incompatible. Got: x and y for positional axis 1",
    ):
        _ = a + b


def test_named_axis_ak_all():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.all(array < 4, axis=0) == ak.all(named_array < 4, axis="x"))
    assert ak.all(ak.all(array < 4, axis=1) == ak.all(named_array < 4, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.all(named_array < 4, axis=0).named_axis
        == ak.all(named_array < 4, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.all(named_array < 4, axis=1).named_axis
        == ak.all(named_array < 4, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.all(named_array < 4, axis=0, keepdims=True).named_axis
        == ak.all(named_array < 4, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.all(named_array < 4, axis=1, keepdims=True).named_axis
        == ak.all(named_array < 4, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.all(named_array < 4, axis=None))


def test_negative_named_axis_ak_all():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.all(array < 4, axis=-2) == ak.all(named_array < 4, axis="x"))
    assert ak.all(ak.all(array < 4, axis=-1) == ak.all(named_array < 4, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.all(named_array < 4, axis=-2).named_axis
        == ak.all(named_array < 4, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.all(named_array < 4, axis=-1).named_axis
        == ak.all(named_array < 4, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.all(named_array < 4, axis=-2, keepdims=True).named_axis
        == ak.all(named_array < 4, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.all(named_array < 4, axis=-1, keepdims=True).named_axis
        == ak.all(named_array < 4, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.all(named_array < 4, axis=None))


def test_named_axis_ak_almost_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(array1, named_axis=("x", "y"))

    assert ak.almost_equal(array1, array2, check_named_axis=False) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.almost_equal(array1, array2, check_named_axis=True) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.almost_equal(named_array1, array1, check_named_axis=False)
    assert ak.almost_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis=("x", "muons"))
    assert ak.almost_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.almost_equal(named_array1, named_array3, check_named_axis=True)


def test_negative_named_axis_ak_almost_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(
        array1, named_axis={"x": -2, "y": -1}
    )

    assert ak.almost_equal(array1, array2, check_named_axis=False) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.almost_equal(array1, array2, check_named_axis=True) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.almost_equal(named_array1, array1, check_named_axis=False)
    assert ak.almost_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis={"x": -2, "z": -1})
    assert ak.almost_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.almost_equal(named_array1, named_array3, check_named_axis=True)


def test_named_axis_ak_angle():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.angle(array) == ak.angle(named_array))

    # check that result axis names are correctly propagated
    assert ak.angle(named_array).named_axis == {"x": 0, "y": 1}


def test_negative_named_axis_ak_angle():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.angle(array) == ak.angle(named_array))

    # check that result axis names are correctly propagated
    assert ak.angle(named_array).named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_any():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.any(array < 4, axis=0) == ak.any(named_array < 4, axis="x"))
    assert ak.all(ak.any(array < 4, axis=1) == ak.any(named_array < 4, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.any(named_array < 4, axis=0).named_axis
        == ak.any(named_array < 4, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.any(named_array < 4, axis=1).named_axis
        == ak.any(named_array < 4, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.any(named_array < 4, axis=0, keepdims=True).named_axis
        == ak.any(named_array < 4, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.any(named_array < 4, axis=1, keepdims=True).named_axis
        == ak.any(named_array < 4, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.all(named_array < 4, axis=None))


def test_negative_named_axis_ak_any():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.any(array < 4, axis=-2) == ak.any(named_array < 4, axis="x"))
    assert ak.all(ak.any(array < 4, axis=-1) == ak.any(named_array < 4, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.any(named_array < 4, axis=-2).named_axis
        == ak.any(named_array < 4, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.any(named_array < 4, axis=-1).named_axis
        == ak.any(named_array < 4, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.any(named_array < 4, axis=-2, keepdims=True).named_axis
        == ak.any(named_array < 4, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.any(named_array < 4, axis=-1, keepdims=True).named_axis
        == ak.any(named_array < 4, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.all(named_array < 4, axis=None))


def test_named_axis_ak_argcartesian():
    one = ak.Array([[1], [2], [3]])
    two = ak.Array([[4, 5]])
    three = ak.Array([[6, 7]])

    named_one = ak.with_named_axis(one, named_axis=("x", "y"))
    named_two = ak.with_named_axis(two, named_axis=("x", "y"))
    named_three = ak.with_named_axis(three, named_axis=("x", "y"))

    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="x", nested=False
    ).named_axis == {"x": 0, "y": 1}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="x", nested=True
    ).named_axis == {"x": 1, "y": 2}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="x", nested=[0]
    ).named_axis == {"x": 1, "y": 2}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="x", nested=[1]
    ).named_axis == {"x": 0, "y": 2}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="x", nested=[0, 1]
    ).named_axis == {"x": 2, "y": 3}


def test_negative_named_axis_ak_argcartesian():
    one = ak.Array([[1], [2], [3]])
    two = ak.Array([[4, 5]])
    three = ak.Array([[6, 7]])

    named_one = ak.with_named_axis(one, named_axis={"x": -2, "y": -1})
    named_two = ak.with_named_axis(two, named_axis={"x": -2, "y": -1})
    named_three = ak.with_named_axis(three, named_axis={"x": -2, "y": -1})

    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="y", nested=False
    ).named_axis == {"x": -1}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="y", nested=True
    ).named_axis == {"x": -2, "y": -1}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="y", nested=[0]
    ).named_axis == {"x": -1}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="y", nested=[1]
    ).named_axis == {"y": -1}
    assert ak.argcartesian(
        [named_one, named_two, named_three], axis="y", nested=[0, 1]
    ).named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_argcombinations():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    assert (
        ak.argcombinations(named_array, 2, axis=0).named_axis == named_array.named_axis
    )
    assert (
        ak.argcombinations(named_array, 2, axis=1).named_axis == named_array.named_axis
    )


def test_negative_named_axis_ak_argcombinations():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    assert (
        ak.argcombinations(named_array, 2, axis=0).named_axis == named_array.named_axis
    )
    assert (
        ak.argcombinations(named_array, 2, axis=1).named_axis == named_array.named_axis
    )


def test_named_axis_ak_argmax():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.argmax(array, axis=0) == ak.argmax(named_array, axis="x"))
    assert ak.all(ak.argmax(array, axis=1) == ak.argmax(named_array, axis="y"))
    assert ak.all(
        ak.argmax(array, axis=0, keepdims=True)
        == ak.argmax(named_array, axis="x", keepdims=True)
    )
    assert ak.all(
        ak.argmax(array, axis=1, keepdims=True)
        == ak.argmax(named_array, axis="y", keepdims=True)
    )
    assert ak.argmax(array, axis=None) == ak.argmax(named_array, axis=None)

    # check that result axis names are correctly propagated
    assert (
        ak.argmax(named_array, axis=0).named_axis
        == ak.argmax(named_array, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.argmax(named_array, axis=1).named_axis
        == ak.argmax(named_array, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.argmax(named_array, axis=0, keepdims=True).named_axis
        == ak.argmax(named_array, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.argmax(named_array, axis=1, keepdims=True).named_axis
        == ak.argmax(named_array, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.argmax(named_array, axis=None))


def test_negative_named_axis_ak_argmax():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.argmax(array, axis=-2) == ak.argmax(named_array, axis="x"))
    assert ak.all(ak.argmax(array, axis=-1) == ak.argmax(named_array, axis="y"))
    assert ak.all(
        ak.argmax(array, axis=-2, keepdims=True)
        == ak.argmax(named_array, axis="x", keepdims=True)
    )
    assert ak.all(
        ak.argmax(array, axis=-1, keepdims=True)
        == ak.argmax(named_array, axis="y", keepdims=True)
    )
    assert ak.argmax(array, axis=None) == ak.argmax(named_array, axis=None)

    # check that result axis names are correctly propagated
    assert (
        ak.argmax(named_array, axis=-2).named_axis
        == ak.argmax(named_array, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.argmax(named_array, axis=-1).named_axis
        == ak.argmax(named_array, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.argmax(named_array, axis=-2, keepdims=True).named_axis
        == ak.argmax(named_array, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.argmax(named_array, axis=-1, keepdims=True).named_axis
        == ak.argmax(named_array, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.argmax(named_array, axis=None))


def test_named_axis_ak_argmin():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.argmin(array, axis=0) == ak.argmin(named_array, axis="x"))
    assert ak.all(ak.argmin(array, axis=1) == ak.argmin(named_array, axis="y"))
    assert ak.all(
        ak.argmin(array, axis=0, keepdims=True)
        == ak.argmin(named_array, axis="x", keepdims=True)
    )
    assert ak.all(
        ak.argmin(array, axis=1, keepdims=True)
        == ak.argmin(named_array, axis="y", keepdims=True)
    )
    assert ak.argmin(array, axis=None) == ak.argmin(named_array, axis=None)

    # check that result axis names are correctly propagated
    assert (
        ak.argmin(named_array, axis=0).named_axis
        == ak.argmin(named_array, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.argmin(named_array, axis=1).named_axis
        == ak.argmin(named_array, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.argmin(named_array, axis=0, keepdims=True).named_axis
        == ak.argmin(named_array, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.argmin(named_array, axis=1, keepdims=True).named_axis
        == ak.argmin(named_array, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.argmin(named_array, axis=None))


def test_negative_named_axis_ak_argmin():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.argmin(array, axis=-2) == ak.argmin(named_array, axis="x"))
    assert ak.all(ak.argmin(array, axis=-1) == ak.argmin(named_array, axis="y"))
    assert ak.all(
        ak.argmin(array, axis=-2, keepdims=True)
        == ak.argmin(named_array, axis="x", keepdims=True)
    )
    assert ak.all(
        ak.argmin(array, axis=-1, keepdims=True)
        == ak.argmin(named_array, axis="y", keepdims=True)
    )
    assert ak.argmin(array, axis=None) == ak.argmin(named_array, axis=None)

    # check that result axis names are correctly propagated
    assert (
        ak.argmin(named_array, axis=-2).named_axis
        == ak.argmin(named_array, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.argmin(named_array, axis=-1).named_axis
        == ak.argmin(named_array, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.argmin(named_array, axis=-2, keepdims=True).named_axis
        == ak.argmin(named_array, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.argmin(named_array, axis=-1, keepdims=True).named_axis
        == ak.argmin(named_array, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.argmin(named_array, axis=None))


def test_named_axis_ak_argsort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.argsort(array, axis=0) == ak.argsort(named_array, axis="x"))
    assert ak.all(ak.argsort(array, axis=1) == ak.argsort(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.argsort(named_array, axis=0).named_axis
        == ak.argsort(named_array, axis="x").named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.argsort(named_array, axis=1).named_axis
        == ak.argsort(named_array, axis="y").named_axis
        == {"x": 0, "y": 1}
    )


def test_negative_named_axis_ak_argsort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.argsort(array, axis=-2) == ak.argsort(named_array, axis="x"))
    assert ak.all(ak.argsort(array, axis=-1) == ak.argsort(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.argsort(named_array, axis=-2).named_axis
        == ak.argsort(named_array, axis="x").named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.argsort(named_array, axis=-1).named_axis
        == ak.argsort(named_array, axis="y").named_axis
        == {"x": -2, "y": -1}
    )


def test_named_axis_ak_array_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(array1, named_axis=("x", "y"))

    assert ak.array_equal(array1, array2, check_named_axis=False) == ak.array_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.array_equal(array1, array2, check_named_axis=True) == ak.array_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.array_equal(named_array1, array1, check_named_axis=False)
    assert ak.array_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis=("x", "z"))
    assert ak.array_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.array_equal(named_array1, named_array3, check_named_axis=True)


def test_negative_named_axis_ak_array_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(
        array1, named_axis={"x": -2, "y": -1}
    )

    assert ak.array_equal(array1, array2, check_named_axis=False) == ak.array_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.array_equal(array1, array2, check_named_axis=True) == ak.array_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.array_equal(named_array1, array1, check_named_axis=False)
    assert ak.array_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis={"x": -2, "z": -1})
    assert ak.array_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.array_equal(named_array1, named_array3, check_named_axis=True)


def test_named_axis_ak_backend():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    assert ak.backend(array) == ak.backend(named_array)


def test_named_axis_ak_broadcast_fields():
    x = ak.Array([{"x": {"y": 1, "z": 2, "w": [1]}}])
    y = ak.Array([{"x": [{"y": 1}]}])

    nx = ak.with_named_axis(x, named_axis=("x", "y"))
    ny = ak.with_named_axis(y, named_axis=("a", "b"))

    na, nb = ak.broadcast_fields(nx, ny)
    assert na.named_axis == {"x": 0, "y": 1}
    assert nb.named_axis == {"a": 0, "b": 1}


def test_named_axis_ak_cartesian():
    one = ak.Array([[1], [2], [3]])
    two = ak.Array([[4, 5]])
    three = ak.Array([[6, 7]])

    named_one = ak.with_named_axis(one, named_axis=("x", "y"))
    named_two = ak.with_named_axis(two, named_axis=("x", "y"))
    named_three = ak.with_named_axis(three, named_axis=("x", "y"))

    assert ak.cartesian(
        [named_one, named_two, named_three], axis="x", nested=False
    ).named_axis == {"x": 0, "y": 1}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="x", nested=True
    ).named_axis == {"x": 1, "y": 2}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="x", nested=[0]
    ).named_axis == {"x": 1, "y": 2}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="x", nested=[1]
    ).named_axis == {"x": 0, "y": 2}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="x", nested=[0, 1]
    ).named_axis == {"x": 2, "y": 3}


def test_negative_named_axis_ak_cartesian():
    one = ak.Array([[1], [2], [3]])
    two = ak.Array([[4, 5]])
    three = ak.Array([[6, 7]])

    named_one = ak.with_named_axis(one, named_axis={"x": -2, "y": -1})
    named_two = ak.with_named_axis(two, named_axis={"x": -2, "y": -1})
    named_three = ak.with_named_axis(three, named_axis={"x": -2, "y": -1})

    assert ak.cartesian(
        [named_one, named_two, named_three], axis="y", nested=False
    ).named_axis == {"x": -1}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="y", nested=True
    ).named_axis == {"x": -2, "y": -1}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="y", nested=[0]
    ).named_axis == {"x": -1}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="y", nested=[1]
    ).named_axis == {"y": -1}
    assert ak.cartesian(
        [named_one, named_two, named_three], axis="y", nested=[0, 1]
    ).named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_categories():
    pyarrow = pytest.importorskip("pyarrow")  # noqa: F841

    array = ak.str.to_categorical([["one", "two"], ["one", "three"], ["one", "four"]])

    named_array = ak.with_named_axis(array, named_axis=("a", "b"))

    assert ak.all(ak.categories(array) == ak.categories(named_array))  # FIX: ufuncs
    assert (
        ak.categories(array).named_axis == ak.categories(named_array).named_axis == {}
    )


def test_named_axis_ak_combinations():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    assert ak.combinations(named_array, 2, axis=0).named_axis == named_array.named_axis
    assert ak.combinations(named_array, 2, axis=1).named_axis == named_array.named_axis


def test_negative_named_axis_ak_combinations():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    assert ak.combinations(named_array, 2, axis=-2).named_axis == named_array.named_axis
    assert ak.combinations(named_array, 2, axis=-1).named_axis == named_array.named_axis


def test_named_axis_ak_concatenate():
    array1 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array3 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array4 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    all_arrays = [array1, array2, array3, array4]

    named_array1 = ak.with_named_axis(array1, named_axis=(None, None))
    named_array2 = ak.with_named_axis(array1, named_axis=(None, "y"))
    named_array3 = ak.with_named_axis(array1, named_axis=("x", None))
    named_array4 = ak.with_named_axis(array1, named_axis=("x", "y"))

    all_named_arrays = [named_array1, named_array2, named_array3, named_array4]

    assert ak.all(
        ak.concatenate(all_arrays, axis=0) == ak.concatenate(all_named_arrays, axis="x")
    )
    assert ak.all(
        ak.concatenate(all_arrays, axis=1) == ak.concatenate(all_named_arrays, axis="y")
    )

    assert ak.concatenate(all_named_arrays, axis="x").named_axis == {"x": 0, "y": 1}
    assert ak.concatenate(all_named_arrays, axis="y").named_axis == {"x": 0, "y": 1}

    with pytest.raises(
        ValueError,
        match="The named axes are incompatible. Got: x and y for positional axis 0",
    ):
        ak.concatenate(
            [
                ak.with_named_axis(array1, named_axis=("x", None)),
                ak.with_named_axis(array2, named_axis=("y", None)),
            ],
            axis=0,
        )

    with pytest.raises(
        ValueError,
        match="The named axes are incompatible. Got: x and y for positional axis 1",
    ):
        ak.concatenate(
            [
                ak.with_named_axis(array1, named_axis=(None, "x")),
                ak.with_named_axis(array2, named_axis=(None, "y")),
            ],
            axis=1,
        )


def test_negative_named_axis_ak_concatenate():
    array1 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array3 = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    array4 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    all_arrays = [array1, array2, array3, array4]

    named_array1 = ak.with_named_axis(array1, named_axis={})
    named_array2 = ak.with_named_axis(array1, named_axis={"y": -1})
    named_array3 = ak.with_named_axis(array1, named_axis={"x": -2})
    named_array4 = ak.with_named_axis(array1, named_axis={"x": -2, "y": -1})

    all_named_arrays = [named_array1, named_array2, named_array3, named_array4]

    assert ak.all(
        ak.concatenate(all_arrays, axis=-2)
        == ak.concatenate(all_named_arrays, axis="x")
    )
    assert ak.all(
        ak.concatenate(all_arrays, axis=-1)
        == ak.concatenate(all_named_arrays, axis="y")
    )

    assert ak.concatenate(all_named_arrays, axis="x").named_axis == {"x": -2, "y": -1}
    assert ak.concatenate(all_named_arrays, axis="y").named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_copy():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    assert ak.copy(named_array).named_axis == {"x": 0, "y": 1}


# def test_named_axis_ak_corr():
#     array_x = ak.Array([[0, 1.1], [3.3, 4.4]])
#     array_y = ak.Array([[0, 1], [3, 4]])

#     named_array_x = ak.with_named_axis(array_x, ("x", "y"))
#     named_array_y = ak.with_named_axis(array_y, ("x", "y"))

#     assert ak.all(
#         ak.corr(array_x, array_y, axis=0)
#         == ak.corr(named_array_x, named_array_y, axis="x")
#     )
#     assert ak.all(
#         ak.corr(array_x, array_y, axis=1)
#         == ak.corr(named_array_x, named_array_y, axis="y")
#     )
#     assert ak.corr(array_x, array_y, axis=None) == ak.corr(
#         named_array_x, named_array_y, axis=None
#     )

#     assert ak.corr(named_array_x, named_array_y, axis="x").named_axis == {"y": 0}
#     assert ak.corr(named_array_x, named_array_y, axis="y").named_axis == {"x": 0}
#     assert not _get_named_axis(ak.corr(named_array_x, named_array_y, axis=None))


def test_named_axis_ak_count():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.count(array, axis=0) == ak.count(named_array, axis="x"))
    assert ak.all(ak.count(array, axis=1) == ak.count(named_array, axis="y"))
    assert ak.count(array, axis=None) == ak.count(named_array, axis=None)

    assert ak.count(named_array, axis="x").named_axis == {"y": 0}
    assert ak.count(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.count(named_array, axis=None))


def test_negative_named_axis_ak_count():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.count(array, axis=-2) == ak.count(named_array, axis="x"))
    assert ak.all(ak.count(array, axis=-1) == ak.count(named_array, axis="y"))
    assert ak.count(array, axis=None) == ak.count(named_array, axis=None)

    assert ak.count(named_array, axis="x").named_axis == {"y": -1}
    assert ak.count(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.count(named_array, axis=None))


def test_named_axis_ak_count_nonzero():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.count_nonzero(array, axis=0) == ak.count_nonzero(named_array, axis="x")
    )
    assert ak.all(
        ak.count_nonzero(array, axis=1) == ak.count_nonzero(named_array, axis="y")
    )
    assert ak.count_nonzero(array, axis=None) == ak.count_nonzero(
        named_array, axis=None
    )

    assert ak.count_nonzero(named_array, axis="x").named_axis == {"y": 0}
    assert ak.count_nonzero(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.count_nonzero(named_array, axis=None))


def test_negative_named_axis_ak_count_nonzero():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(
        ak.count_nonzero(array, axis=-2) == ak.count_nonzero(named_array, axis="x")
    )
    assert ak.all(
        ak.count_nonzero(array, axis=-1) == ak.count_nonzero(named_array, axis="y")
    )
    assert ak.count_nonzero(array, axis=None) == ak.count_nonzero(
        named_array, axis=None
    )

    assert ak.count_nonzero(named_array, axis="x").named_axis == {"y": -1}
    assert ak.count_nonzero(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.count_nonzero(named_array, axis=None))


# def test_named_axis_ak_covar():
#     array_x = ak.Array([[0, 1.1], [3.3, 4.4]])
#     array_y = ak.Array([[0, 1], [3, 4]])

#     named_array_x = ak.with_named_axis(array_x, ("x", "y"))
#     named_array_y = ak.with_named_axis(array_y, ("x", "y"))

#     assert ak.all(
#         ak.covar(array_x, array_y, axis=0)
#         == ak.covar(named_array_x, named_array_y, axis="x")
#     )
#     assert ak.all(
#         ak.covar(array_x, array_y, axis=1)
#         == ak.covar(named_array_x, named_array_y, axis="y")
#     )
#     assert ak.covar(array_x, array_y, axis=None) == ak.covar(
#         named_array_x, named_array_y, axis=None
#     )

#     assert ak.covar(named_array_x, named_array_y, axis="x").named_axis == {"y": 0}
#     assert ak.covar(named_array_x, named_array_y, axis="y").named_axis == {"x": 0}
#     assert not _get_named_axis(ak.covar(named_array_x, named_array_y, axis=None))


def test_named_axis_ak_drop_none():
    array = ak.Array([[1, None], [3], [None], [4, None, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.drop_none(array, axis=0) == ak.drop_none(named_array, axis="x"))
    assert ak.all(ak.drop_none(array, axis=1) == ak.drop_none(named_array, axis="y"))
    assert ak.all(
        ak.drop_none(array, axis=None) == ak.drop_none(named_array, axis=None)
    )

    assert ak.drop_none(named_array, axis="x").named_axis == {"x": 0, "y": 1}
    assert ak.drop_none(named_array, axis="y").named_axis == {"x": 0, "y": 1}
    assert ak.drop_none(named_array, axis=None).named_axis == {"x": 0, "y": 1}


def test_negative_named_axis_ak_drop_none():
    array = ak.Array([[1, None], [3], [None], [4, None, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.drop_none(array, axis=-2) == ak.drop_none(named_array, axis="x"))
    assert ak.all(ak.drop_none(array, axis=-1) == ak.drop_none(named_array, axis="y"))
    assert ak.all(
        ak.drop_none(array, axis=None) == ak.drop_none(named_array, axis=None)
    )

    assert ak.drop_none(named_array, axis="x").named_axis == {"x": -2, "y": -1}
    assert ak.drop_none(named_array, axis="y").named_axis == {"x": -2, "y": -1}
    assert ak.drop_none(named_array, axis=None).named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_enforce_type():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.enforce_type(named_array, "var * ?int64").named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_fill_none():
    array = ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.fill_none(array, 0, axis=0) == ak.fill_none(named_array, 0, axis="x")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=1) == ak.fill_none(named_array, 0, axis="y")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=None) == ak.fill_none(named_array, 0, axis=None)
    )

    assert ak.fill_none(named_array, 0, axis="x").named_axis == {"x": 0, "y": 1}
    assert ak.fill_none(named_array, 0, axis="y").named_axis == {"x": 0, "y": 1}
    assert ak.fill_none(named_array, 0, axis=None).named_axis == {"x": 0, "y": 1}


def test_negative_named_axis_ak_fill_none():
    array = ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(
        ak.fill_none(array, 0, axis=-2) == ak.fill_none(named_array, 0, axis="x")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=-1) == ak.fill_none(named_array, 0, axis="y")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=None) == ak.fill_none(named_array, 0, axis=None)
    )

    assert ak.fill_none(named_array, 0, axis="x").named_axis == {"x": -2, "y": -1}
    assert ak.fill_none(named_array, 0, axis="y").named_axis == {"x": -2, "y": -1}
    assert ak.fill_none(named_array, 0, axis=None).named_axis == {"x": -2, "y": -1}


def test_named_axis_ak_firsts():
    array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.firsts(array, axis=0) == ak.firsts(named_array, axis="x"))
    assert ak.all(ak.firsts(array, axis=1) == ak.firsts(named_array, axis="y"))

    assert ak.firsts(named_array, axis="x").named_axis == {"y": 0}
    assert ak.firsts(named_array, axis="y").named_axis == {"x": 0}


def test_negative_named_axis_ak_firsts():
    array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.firsts(array, axis=-2) == ak.firsts(named_array, axis="x"))
    assert ak.all(ak.firsts(array, axis=-1) == ak.firsts(named_array, axis="y"))

    assert ak.firsts(named_array, axis="x").named_axis == {"y": -1}
    assert ak.firsts(named_array, axis="y").named_axis == {"x": -1}


def test_named_axis_ak_flatten():
    array = ak.Array([[[1.1, 2.2]], [[]], [[3.3]], [[]], [[]], [[4.4, 5.5]]])

    named_array = ak.with_named_axis(array, ("x", "y", "z"))

    assert ak.all(ak.flatten(array, axis=0) == ak.flatten(named_array, axis="x"))
    assert ak.all(ak.flatten(array, axis=1) == ak.flatten(named_array, axis="y"))
    assert ak.all(ak.flatten(array, axis=2) == ak.flatten(named_array, axis="z"))
    assert ak.all(ak.flatten(array, axis=None) == ak.flatten(named_array, axis=None))

    assert ak.flatten(named_array, axis="x").named_axis == {"x": 0, "y": 1, "z": 2}
    assert ak.flatten(named_array, axis="y").named_axis == {"x": 0, "z": 1}
    assert ak.flatten(named_array, axis="z").named_axis == {"x": 0, "y": 1}
    assert not _get_named_axis(ak.flatten(named_array, axis=None))


def test_negative_named_axis_ak_flatten():
    array = ak.Array([[[1.1, 2.2]], [[]], [[3.3]], [[]], [[]], [[4.4, 5.5]]])

    named_array = ak.with_named_axis(array, named_axis={"x": -3, "y": -2, "z": -1})

    assert ak.all(ak.flatten(array, axis=-3) == ak.flatten(named_array, axis="x"))
    assert ak.all(ak.flatten(array, axis=-2) == ak.flatten(named_array, axis="y"))
    assert ak.all(ak.flatten(array, axis=-1) == ak.flatten(named_array, axis="z"))
    assert ak.all(ak.flatten(array, axis=None) == ak.flatten(named_array, axis=None))

    assert ak.flatten(named_array, axis="x").named_axis == {"x": -3, "y": -2, "z": -1}
    assert ak.flatten(named_array, axis="y").named_axis == {"x": -2, "z": -1}
    assert ak.flatten(named_array, axis="z").named_axis == {"x": -2, "y": -1}
    assert not _get_named_axis(ak.flatten(named_array, axis=None))


def test_named_axis_ak_imag():
    array = ak.Array([[1 + 2j], [2 + 1j], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.imag(array) == ak.imag(named_array))
    assert ak.imag(named_array).named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_is_none():
    array = ak.Array([[[1, None]], [[3]], [[None]], [[4, None, 6]]])

    named_array = ak.with_named_axis(array, ("x", "y", "z"))

    assert ak.all(ak.is_none(array, axis=0) == ak.is_none(named_array, axis="x"))
    assert ak.all(ak.is_none(array, axis=1) == ak.is_none(named_array, axis="y"))
    assert ak.all(ak.is_none(array, axis=2) == ak.is_none(named_array, axis="z"))

    assert ak.is_none(named_array, axis="x").named_axis == {"x": 0}
    assert ak.is_none(named_array, axis="y").named_axis == {"x": 0, "y": 1}
    assert ak.is_none(named_array, axis="z").named_axis == {"x": 0, "y": 1, "z": 2}


def test_negative_named_axis_ak_is_none():
    array = ak.Array([[[1, None]], [[3]], [[None]], [[4, None, 6]]])

    named_array = ak.with_named_axis(array, named_axis={"x": -3, "y": -2, "z": -1})

    assert ak.all(ak.is_none(array, axis=-3) == ak.is_none(named_array, axis="x"))
    assert ak.all(ak.is_none(array, axis=-2) == ak.is_none(named_array, axis="y"))
    assert ak.all(ak.is_none(array, axis=-1) == ak.is_none(named_array, axis="z"))

    assert ak.is_none(named_array, axis="x").named_axis == {"z": -1}
    assert ak.is_none(named_array, axis="y").named_axis == {"y": -2, "z": -1}
    assert ak.is_none(named_array, axis="z").named_axis == {"x": -3, "y": -2, "z": -1}


def test_named_axis_ak_isclose():
    a = b = ak.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )

    na = ak.with_named_axis(a, ("x", "y", "z"))
    nb = ak.with_named_axis(b, ("x", "y", "z"))
    assert ak.all(ak.isclose(a, b) == ak.isclose(na, nb))

    na = ak.with_named_axis(a, (None, "y", "z"))
    nb = ak.with_named_axis(b, ("x", "y", None))
    assert ak.isclose(na, nb).named_axis == {"x": 0, "y": 1, "z": 2}


def test_named_axis_ak_local_index():
    array = ak.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )

    named_array = ak.with_named_axis(array, ("x", "y", "z"))

    assert ak.all(
        ak.local_index(array, axis=0) == ak.local_index(named_array, axis="x")
    )
    assert ak.all(
        ak.local_index(array, axis=1) == ak.local_index(named_array, axis="y")
    )
    assert ak.all(
        ak.local_index(array, axis=2) == ak.local_index(named_array, axis="z")
    )

    assert ak.local_index(named_array, axis="x").named_axis == {"x": 0}
    assert ak.local_index(named_array, axis="y").named_axis == {"x": 0, "y": 1}
    assert ak.local_index(named_array, axis="z").named_axis == {"x": 0, "y": 1, "z": 2}


def test_negative_named_axis_ak_local_index():
    array = ak.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )
    named_array = ak.with_named_axis(array, {"x": -3, "y": -2, "z": -1})

    assert ak.local_index(named_array, axis="x").named_axis == {"z": -1}
    assert ak.local_index(named_array, axis="y").named_axis == {"y": -2, "z": -1}
    assert ak.local_index(named_array, axis="z").named_axis == {
        "x": -3,
        "y": -2,
        "z": -1,
    }


def test_named_axis_ak_mask():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    mask = array > 3

    named_array = ak.with_named_axis(array, ("x", "y"))
    named_mask = named_array > 3

    assert ak.all(ak.mask(array, mask) == ak.mask(named_array, mask))
    assert ak.all(ak.mask(array, mask) == ak.mask(named_array, named_mask))

    assert ak.mask(named_array, mask).named_axis == named_array.named_axis
    assert ak.mask(named_array, named_mask).named_axis == named_array.named_axis


def test_named_axis_ak_max():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.max(array, axis=0) == ak.max(named_array, axis="x"))
    assert ak.all(ak.max(array, axis=1) == ak.max(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.max(named_array, axis=0).named_axis
        == ak.max(named_array, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.max(named_array, axis=1).named_axis
        == ak.max(named_array, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.max(named_array, axis=0, keepdims=True).named_axis
        == ak.max(named_array, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.max(named_array, axis=1, keepdims=True).named_axis
        == ak.max(named_array, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.max(named_array, axis=None))


def test_negative_named_axis_ak_max():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.max(array, axis=-2) == ak.max(named_array, axis="x"))
    assert ak.all(ak.max(array, axis=-1) == ak.max(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.max(named_array, axis=-2).named_axis
        == ak.max(named_array, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.max(named_array, axis=-1).named_axis
        == ak.max(named_array, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.max(named_array, axis=-2, keepdims=True).named_axis
        == ak.max(named_array, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.max(named_array, axis=-1, keepdims=True).named_axis
        == ak.max(named_array, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.max(named_array, axis=None))


def test_named_axis_ak_mean():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.mean(array, axis=0) == ak.mean(named_array, axis="x"))
    assert ak.all(ak.mean(array, axis=1) == ak.mean(named_array, axis="y"))
    assert ak.mean(array, axis=None) == ak.mean(named_array, axis=None)

    assert ak.mean(named_array, axis="x").named_axis == {"y": 0}
    assert ak.mean(named_array, axis="y").named_axis == {"x": 0}
    assert ak.mean(named_array, axis="x", keepdims=True).named_axis == {"x": 0, "y": 1}
    assert ak.mean(named_array, axis="y", keepdims=True).named_axis == {"x": 0, "y": 1}
    assert not _get_named_axis(ak.mean(named_array, axis=None))


def test_negative_named_axis_ak_mean():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    assert ak.all(ak.mean(array, axis=-2) == ak.mean(named_array, axis="x"))
    assert ak.all(ak.mean(array, axis=-1) == ak.mean(named_array, axis="y"))
    assert ak.mean(array, axis=None) == ak.mean(named_array, axis=None)

    assert ak.mean(named_array, axis="x").named_axis == {"y": -1}
    assert ak.mean(named_array, axis="y").named_axis == {"x": -1}
    assert ak.mean(named_array, axis="x", keepdims=True).named_axis == {
        "x": -2,
        "y": -1,
    }
    assert ak.mean(named_array, axis="y", keepdims=True).named_axis == {
        "x": -2,
        "y": -1,
    }
    assert not _get_named_axis(ak.mean(named_array, axis=None))


def test_named_axis_ak_merge_option_of_records():
    array = ak.Array([None, {"a": 1}, {"a": 2}])

    named_array = ak.with_named_axis(array, named_axis=("x",))

    assert (
        ak.merge_option_of_records(named_array, axis="x").named_axis
        == named_array.named_axis
    )


def test_named_axis_ak_merge_union_of_records():
    array = ak.concatenate(([{"a": 1}], [{"b": 2}]))

    named_array = ak.with_named_axis(array, named_axis=("x",))

    assert (
        ak.merge_union_of_records(named_array, axis="x").named_axis
        == named_array.named_axis
    )


def test_named_axis_ak_min():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.min(array, axis=0) == ak.min(named_array, axis="x"))
    assert ak.all(ak.min(array, axis=1) == ak.min(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.min(named_array, axis=0).named_axis
        == ak.min(named_array, axis="x").named_axis
        == {"y": 0}
    )
    assert (
        ak.min(named_array, axis=1).named_axis
        == ak.min(named_array, axis="y").named_axis
        == {"x": 0}
    )
    assert (
        ak.min(named_array, axis=0, keepdims=True).named_axis
        == ak.min(named_array, axis="x", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.min(named_array, axis=1, keepdims=True).named_axis
        == ak.min(named_array, axis="y", keepdims=True).named_axis
        == {"x": 0, "y": 1}
    )
    assert not _get_named_axis(ak.min(named_array, axis=None))


def test_negative_named_axis_ak_min():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis={"x": -2, "y": -1})

    # first check that they work the same
    assert ak.all(ak.min(array, axis=-2) == ak.min(named_array, axis="x"))
    assert ak.all(ak.min(array, axis=-1) == ak.min(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.min(named_array, axis=-2).named_axis
        == ak.min(named_array, axis="x").named_axis
        == {"y": -1}
    )
    assert (
        ak.min(named_array, axis=-1).named_axis
        == ak.min(named_array, axis="y").named_axis
        == {"x": -1}
    )
    assert (
        ak.min(named_array, axis=-2, keepdims=True).named_axis
        == ak.min(named_array, axis="x", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert (
        ak.min(named_array, axis=-1, keepdims=True).named_axis
        == ak.min(named_array, axis="y", keepdims=True).named_axis
        == {"x": -2, "y": -1}
    )
    assert not _get_named_axis(ak.min(named_array, axis=None))


def test_named_axis_ak_moment():
    array = ak.Array([[0, 1.1], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.moment(array, 0, axis=0) == ak.moment(named_array, 0, axis="x"))
    assert ak.all(ak.moment(array, 0, axis=1) == ak.moment(named_array, 0, axis="y"))
    assert ak.moment(array, 0, axis=None) == ak.moment(named_array, 0, axis=None)

    assert ak.moment(named_array, 0, axis="x").named_axis == {"y": 0}
    assert ak.moment(named_array, 0, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.moment(named_array, 0, axis=None))


def test_negative_named_axis_ak_moment():
    array = ak.Array([[0, 1.1], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.moment(array, 0, axis=-2) == ak.moment(named_array, 0, axis="x"))
    assert ak.all(ak.moment(array, 0, axis=-1) == ak.moment(named_array, 0, axis="y"))
    assert ak.moment(array, 0, axis=None) == ak.moment(named_array, 0, axis=None)

    assert ak.moment(named_array, 0, axis="x").named_axis == {"y": -1}
    assert ak.moment(named_array, 0, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.moment(named_array, 0, axis=None))


def test_named_axis_ak_nan_to_none():
    array = ak.Array([[0, np.nan], [np.nan], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.nan_to_none(array) == ak.nan_to_none(named_array))
    assert ak.nan_to_none(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_nan_to_num():
    array = ak.Array([[0, np.nan], [np.nan], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.nan_to_num(array, nan=0.0) == ak.nan_to_num(named_array, nan=0.0))
    assert ak.nan_to_num(named_array, nan=0.0).named_axis == named_array.named_axis


def test_named_axis_ak_num():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.num(array, axis=0) == ak.num(named_array, axis="x")
    assert ak.all(ak.num(array, axis=1) == ak.num(named_array, axis="y"))

    assert ak.num(named_array, axis="y").named_axis == {"y": 0}
    assert not _get_named_axis(ak.num(named_array, axis="x"))


def test_negative_named_axis_ak_num():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.num(array, axis=-2) == ak.num(named_array, axis="x")
    assert ak.all(ak.num(array, axis=-1) == ak.num(named_array, axis="y"))

    assert ak.num(named_array, axis="y").named_axis == {"y": 0}
    assert not _get_named_axis(ak.num(named_array, axis="x"))


def test_named_axis_ak_ones_like():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ones_like(array) == ak.ones_like(named_array))

    assert ak.ones_like(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_pad_none():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.pad_none(array, 3, axis=0) == ak.pad_none(named_array, 3, axis=0))
    assert ak.all(ak.pad_none(array, 3, axis=1) == ak.pad_none(named_array, 3, axis=1))

    assert ak.pad_none(named_array, 3, axis=0).named_axis == named_array.named_axis
    assert ak.pad_none(named_array, 3, axis=1).named_axis == named_array.named_axis


def test_named_axis_ak_prod():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.prod(array, axis=0) == ak.prod(named_array, axis="x"))
    assert ak.all(ak.prod(array, axis=1) == ak.prod(named_array, axis="y"))
    assert ak.prod(array, axis=None) == ak.prod(named_array, axis=None)

    assert ak.prod(named_array, axis="x").named_axis == {"y": 0}
    assert ak.prod(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.prod(named_array, axis=None))


def test_negative_named_axis_ak_prod():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.prod(array, axis=-2) == ak.prod(named_array, axis="x"))
    assert ak.all(ak.prod(array, axis=-1) == ak.prod(named_array, axis="y"))
    assert ak.prod(array, axis=None) == ak.prod(named_array, axis=None)

    assert ak.prod(named_array, axis="x").named_axis == {"y": -1}
    assert ak.prod(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.prod(named_array, axis=None))


def test_named_axis_ak_ptp():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ptp(array, axis=0) == ak.ptp(named_array, axis="x"))
    assert ak.all(ak.ptp(array, axis=1) == ak.ptp(named_array, axis="y"))
    assert ak.ptp(array, axis=None) == ak.ptp(named_array, axis=None)

    assert ak.ptp(named_array, axis="x").named_axis == {"y": 0}
    assert ak.ptp(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.ptp(named_array, axis=None))


def test_negative_named_axis_ak_ptp():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.ptp(array, axis=-2) == ak.ptp(named_array, axis="x"))
    assert ak.all(ak.ptp(array, axis=-1) == ak.ptp(named_array, axis="y"))
    assert ak.ptp(array, axis=None) == ak.ptp(named_array, axis=None)

    assert ak.ptp(named_array, axis="x").named_axis == {"y": -1}
    assert ak.ptp(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.ptp(named_array, axis=None))


def test_named_axis_ak_ravel():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ravel(array) == ak.ravel(named_array))

    assert not _get_named_axis(ak.ravel(named_array))


def test_named_axis_ak_real():
    array = ak.Array([[1 + 2j], [2 + 1j], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.real(array) == ak.real(named_array))
    assert ak.real(named_array).named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_round():
    array = ak.Array([[1.234], [2.345, 3.456], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.round(array) == ak.round(named_array))
    assert ak.round(named_array).named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_run_lengths():
    array = ak.Array([[1.1, 1.1, 1.1, 2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.run_lengths(array) == ak.run_lengths(named_array))

    assert ak.run_lengths(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_singletons():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.singletons(array, axis=0) == ak.singletons(named_array, axis="x"))
    assert ak.all(ak.singletons(array, axis=1) == ak.singletons(named_array, axis="y"))

    assert ak.singletons(named_array, axis=0).named_axis == {"x": 0, "y": 2}
    assert ak.singletons(named_array, axis=1).named_axis == {"x": 0, "y": 1}


def test_negative_named_axis_ak_singletons():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.singletons(array, axis=-2) == ak.singletons(named_array, axis="x"))
    assert ak.all(ak.singletons(array, axis=-1) == ak.singletons(named_array, axis="y"))

    assert ak.singletons(named_array, axis=-2).named_axis == {"x": -3, "y": -1}
    assert ak.singletons(named_array, axis=-1).named_axis == {"x": -3, "y": -2}


def test_named_axis_ak_softmax():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.softmax(array, axis=-1) == ak.softmax(named_array, axis="y"))

    assert ak.softmax(named_array, axis="y").named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_sort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("x", "y"))

    # first check that they work the same
    assert ak.all(ak.sort(array, axis=0) == ak.sort(named_array, axis="x"))
    assert ak.all(ak.sort(array, axis=1) == ak.sort(named_array, axis="y"))

    # check that result axis names are correctly propagated
    assert (
        ak.sort(named_array, axis=0).named_axis
        == ak.sort(named_array, axis="x").named_axis
        == {"x": 0, "y": 1}
    )
    assert (
        ak.sort(named_array, axis=1).named_axis
        == ak.sort(named_array, axis="y").named_axis
        == {"x": 0, "y": 1}
    )


def test_named_axis_ak_std():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.std(array, axis=0) == ak.std(named_array, axis="x"))
    assert ak.all(ak.std(array, axis=1) == ak.std(named_array, axis="y"))
    assert ak.std(array, axis=None) == ak.std(named_array, axis=None)

    assert ak.std(named_array, axis="x").named_axis == {"y": 0}
    assert ak.std(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.std(named_array, axis=None))


def test_negative_named_axis_ak_std():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.std(array, axis=-2) == ak.std(named_array, axis="x"))
    assert ak.all(ak.std(array, axis=-1) == ak.std(named_array, axis="y"))
    assert ak.std(array, axis=None) == ak.std(named_array, axis=None)

    assert ak.std(named_array, axis="x").named_axis == {"y": -1}
    assert ak.std(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.std(named_array, axis=None))


def test_named_axis_ak_strings_astype():
    array = ak.Array([["1", "2"], ["3"], ["4", "5", "6"]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.strings_astype(array, np.int32) == ak.strings_astype(named_array, np.int32)
    )

    assert ak.strings_astype(named_array, np.int32).named_axis == named_array.named_axis


def test_named_axis_ak_sum():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.sum(array, axis=0) == ak.sum(named_array, axis="x"))
    assert ak.all(ak.sum(array, axis=1) == ak.sum(named_array, axis="y"))
    assert ak.sum(array, axis=None) == ak.sum(named_array, axis=None)

    assert ak.sum(named_array, axis="x").named_axis == {"y": 0}
    assert ak.sum(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.sum(named_array, axis=None))


def test_negative_named_axis_ak_sum():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.sum(array, axis=-2) == ak.sum(named_array, axis="x"))
    assert ak.all(ak.sum(array, axis=-1) == ak.sum(named_array, axis="y"))
    assert ak.sum(array, axis=None) == ak.sum(named_array, axis=None)

    assert ak.sum(named_array, axis="x").named_axis == {"y": -1}
    assert ak.sum(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.sum(named_array, axis=None))


def test_named_axis_ak_to_backend():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.to_backend(named_array, "typetracer").named_axis == named_array.named_axis


def test_named_axis_ak_to_packed():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.to_packed(array) == ak.to_packed(named_array))

    assert ak.to_packed(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_unflatten():
    array = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    counts = ak.Array([2, 2, 1, 2, 1, 1])

    assert ak.all(
        ak.unflatten(array, counts, axis=1)
        == ak.unflatten(named_array, counts, axis="y")
    )
    assert not _get_named_axis(ak.unflatten(named_array, counts, axis="y"))


def test_named_axis_ak_unzip():
    array = ak.Array(
        [
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ]
    )
    named_array = ak.with_named_axis(array, ("x", "y"))
    x, y = ak.unzip(named_array)
    assert x.named_axis == y.named_axis == {"x": 0, "y": 1}


def test_named_axis_ak_values_astype():
    array = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.values_astype(array, np.float32) == ak.values_astype(named_array, np.float32)
    )

    assert (
        ak.values_astype(named_array, np.float32).named_axis == named_array.named_axis
    )


def test_named_axis_ak_var():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.var(array, axis=0) == ak.var(named_array, axis="x"))
    assert ak.all(ak.var(array, axis=1) == ak.var(named_array, axis="y"))
    assert ak.var(array, axis=None) == ak.var(named_array, axis=None)

    assert ak.var(named_array, axis="x").named_axis == {"y": 0}
    assert ak.var(named_array, axis="y").named_axis == {"x": 0}
    assert not _get_named_axis(ak.var(named_array, axis=None))


def test_negative_named_axis_ak_var():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, {"x": -2, "y": -1})

    assert ak.all(ak.var(array, axis=-2) == ak.var(named_array, axis="x"))
    assert ak.all(ak.var(array, axis=-1) == ak.var(named_array, axis="y"))
    assert ak.var(array, axis=None) == ak.var(named_array, axis=None)

    assert ak.var(named_array, axis="x").named_axis == {"y": -1}
    assert ak.var(named_array, axis="y").named_axis == {"x": -1}
    assert not _get_named_axis(ak.var(named_array, axis=None))


def test_named_axis_ak_where():
    a = ak.Array([[1, 2], [3, 4]])
    na = ak.with_named_axis(a, ("x", "y"))

    assert ak.all(ak.where(a > 2, 0, 1) == ak.where(na > 2, 0, 1))
    assert ak.where(na > 2, 0, 1).named_axis == {"x": 0, "y": 1}
    assert ak.where(na > 2, na, 1).named_axis == {"x": 0, "y": 1}

    nb = ak.with_named_axis(a, ("a", "b"))
    with pytest.raises(ValueError):
        _ = ak.where(na > 2, nb, 1)


def test_named_axis_ak_with_field():
    array = ak.Array(
        [
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ]
    )
    named_array = ak.with_named_axis(array, ("x", "y"))
    xyz = ak.with_field(named_array, ak.Array([[1], [2], [3]]), "z")
    x, y, z = ak.unzip(xyz)
    assert x.named_axis == y.named_axis == z.named_axis == {"x": 0, "y": 1}

    named_z = ak.with_named_axis(ak.Array([[1], [2], [3]]), ("a", "b"))
    with pytest.raises(ValueError):
        ak.with_field(named_array, named_z, "z")


def test_named_axis_ak_with_name():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.with_name(named_array, "new_name").named_axis == named_array.named_axis


def test_named_axis_ak_with_named_axis():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    # tuple
    named_array = ak.with_named_axis(array, ("x", "y"))
    assert named_array.named_axis == {"x": 0, "y": 1}

    # dict
    named_array = ak.with_named_axis(array, {"x": 0, "y": -1})
    assert named_array.named_axis == {"x": 0, "y": -1}


def test_named_axis_ak_with_parameter():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert (
        ak.with_parameter(named_array, "param", 1.0).named_axis
        == named_array.named_axis
    )


def test_named_axis_ak_without_parameters():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    named_array_with_parameteter = ak.with_parameter(named_array, "param", 1.0)

    assert (
        ak.without_parameters(named_array_with_parameteter).named_axis
        == named_array.named_axis
    )


def test_named_axis_ak_zeros_like():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.zeros_like(array) == ak.zeros_like(named_array))

    assert ak.zeros_like(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_zip():
    named_array1 = ak.with_named_axis(ak.Array([1, 2, 3]), ("x",))
    named_array2 = ak.with_named_axis(ak.Array([[4, 5, 6], [], [7]]), ("x", "y"))

    assert ak.zip({"x": named_array1, "y": named_array2}).named_axis == {"x": 0, "y": 1}

    named_array1 = ak.with_named_axis(ak.Array([1, 2, 3]), ("a",))
    named_array2 = ak.with_named_axis(ak.Array([[4, 5, 6], [], [7]]), ("x", "y"))

    with pytest.raises(ValueError):
        _ = ak.zip({"x": named_array1, "y": named_array2})
