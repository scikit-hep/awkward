from __future__ import annotations

import awkward._typing as tp
from awkward._regularize import is_integer

if tp.TYPE_CHECKING:
    from awkward._layout import HighLevelContext
    from awkward.highlevel import Array


# axis names are hashables, mostly strings,
# except for integers, which are reserved for positional axis.
AxisName: tp.TypeAlias = tp.Hashable

AxisMapping = tp.Mapping[AxisName, int]  # e.g.: {"x": 0, "y": 1, "z": 2}
AxisTuple = tuple[AxisName, ...]  # e.g.: ("x", "y", None) where None is a wildcard

_NamedAxisKey: str = "__named_axis__"


def _supports_named_axis(array: Array | HighLevelContext) -> bool:
    """Check if the given array supports named axis.

    Args:
        array (Array): The array to check.

    Returns:
        bool: True if the array supports named axis, False otherwise.
    """
    return bool((getattr(array, "attrs", {}) or {}).get(_NamedAxisKey, {}))


class TmpNamedAxisMarker:
    """
    The TmpNamedAxisMarker class serves as a placeholder for axis wildcards. It is used temporarily during
    the process of axis manipulation and conversion. This marker helps in identifying the positions
    in the axis tuple that are yet to be assigned a specific axis name or value.
    """


def _is_valid_named_axis(axis: AxisName) -> bool:
    return isinstance(axis, AxisName) and not is_integer(axis)


def _check_valid_axis(axis: AxisName) -> AxisName:
    if not _is_valid_named_axis(axis):
        raise ValueError(f"Axis names must be hashable and not int, got {axis!r}")
    return axis


def _check_axis_mapping_unique_values(axis_mapping: AxisMapping) -> None:
    if len(set(axis_mapping.values())) != len(axis_mapping):
        raise ValueError(
            f"Named axis mapping must be unique for each positional axis, got: {axis_mapping}"
        )


def _axis_tuple_to_mapping(axis_tuple: AxisTuple) -> AxisMapping:
    """
    Converts a tuple of axis names to a dictionary mapping axis names to their positions.

    Args:
        axis_tuple (AxisTuple): A tuple of axis names. Can include None as a wildcard.

    Returns:
        AxisMapping: A dictionary mapping axis names to their positions.

    Examples:
        >>> _axis_tuple_to_mapping(("x", "y", None))
        {"x": 0, "y": 1, TmpNamedAxisMarker(): 2}
    """
    return {
        (_check_valid_axis(axis) if axis is not None else TmpNamedAxisMarker()): i
        for i, axis in enumerate(axis_tuple)
    }


def _axis_mapping_to_tuple(axis_mapping: AxisMapping) -> AxisTuple:
    """
    Converts a dictionary mapping of axis names to their positions to a tuple of axis names.
    Does not allow the same values to be repeated in the mapping.

    Args:
        axis_mapping (AxisMapping): A dictionary mapping axis names to their positions.

    Returns:
        AxisTuple: A tuple of axis names. None is used as a placeholder for TmpNamedAxisMarker.

    Examples:
        >>> _axis_mapping_to_tuple({"x": 0, "y": 1, TmpNamedAxisMarker(): 2})
        ("x", "y", None)
        >>> _axis_mapping_to_tuple({"x": 0, "y": -1, TmpNamedAxisMarker(): 1})
        ("x", None, "y")
        >>> _axis_mapping_to_tuple({"x": 0, "y": 0, TmpNamedAxisMarker(): 1})
        Traceback (most recent call last):
        ...
        ValueError: Axis positions must be unique, got: {"x": 0, "y": 0, TmpNamedAxisMarker(): 1}
    """
    _check_axis_mapping_unique_values(axis_mapping)

    axis_list: list[AxisName | None] = [None] * len(axis_mapping)
    for ax, pos in axis_mapping.items():
        if isinstance(ax, TmpNamedAxisMarker):
            axis_list[pos] = None
        else:
            axis_list[pos] = _check_valid_axis(ax)
    return tuple(axis_list)


def _any_axis_to_positional_axis(
    axis: AxisName | AxisTuple,
    named_axis: AxisTuple,
    positional_axis: tuple[int, ...],
) -> AxisTuple | int | None:
    """
    Converts any axis (int, AxisName, AxisTuple, or None) to a positional axis (int or AxisTuple).

    Args:
        axis (int | AxisName | AxisTuple | None): The axis to convert. Can be an integer, an AxisName, an AxisTuple, or None.
        named_axis (AxisTuple): The named axis mapping to use for conversion.
        positional_axis (tuple[int, ...]): The positional axis mapping to use for conversion.

    Returns:
        int | AxisTuple | None: The converted axis. Will be an integer, an AxisTuple, or None.

    Raises:
        ValueError: If the axis is not found in the named axis mapping.

    Examples:
        >>> _any_axis_to_positional_axis("x", ("x", "y", "z"), (0, 1, 2))
        0
        >>> _any_axis_to_positional_axis(("x", "z"), ("x", "y", "z"), (0, 1, 2))
        (0, 2)
    """
    if isinstance(axis, (tuple, list)):
        return tuple(
            _one_axis_to_positional_axis(ax, named_axis, positional_axis) for ax in axis
        )
    else:
        return _one_axis_to_positional_axis(axis, named_axis, positional_axis)


def _one_axis_to_positional_axis(
    axis: AxisName | None,
    named_axis: AxisTuple,
    positional_axis: tuple[int, ...],
) -> int | None:
    """
    Converts a single axis (int, AxisName, or None) to a positional axis (int or None).

    Args:
        axis (int | AxisName | None): The axis to convert. Can be an integer, an AxisName, or None.
        named_axis (AxisTuple): The named axis mapping to use for conversion.
        positional_axis (tuple[int, ...]): The positional axis mapping to use for conversion.

    Returns:
        int | None: The converted axis. Will be an integer or None.

    Raises:
        ValueError: If the axis is not found in the named axis mapping.

    Examples:
        >>> _one_axis_to_positional_axis("x", ("x", "y", "z"), (0, 1, 2))
        0
    """
    if isinstance(axis, int) or axis is None:
        return axis
    elif axis in named_axis:
        return positional_axis[named_axis.index(axis)]
    else:
        raise ValueError(f"Invalid axis '{axis}'")


def _set_named_axis_to_attrs(
    attrs: tp.Mapping,
    named_axis: AxisTuple | AxisMapping,
    overwrite: bool = True,
) -> tp.Mapping:
    """
    Sets the named axis mapping into the given attributes dictionary.

    Args:
        attrs (dict): The attributes dictionary to set the named axis mapping into.
        named_axis (AxisTuple | AxisMapping): The named axis mapping to set. Can be a tuple or a dictionary.
        overwrite (bool, optional): If True, any existing named axis mapping in the attributes dictionary will be overwritten.
            If False, a KeyError will be raised if a named axis mapping already exists in the attributes dictionary.
            Default is True.

    Returns:
        dict: The attributes dictionary with the named axis mapping set.

    Raises:
        TypeError: If the named axis is not a tuple or a dictionary.
        KeyError: If a named axis mapping already exists in the attributes dictionary and overwrite is False.

    Examples:
        >>> attrs = {"other_key": "other_value"}
        >>> named_axis = ("x", "y", "z")
        >>> _set_named_axis_to_attrs(attrs, named_axis)
        {"other_key": "other_value", "__named_axis__": {"x": 0, "y": 1, "z": 2}}
    """
    attrs = dict(attrs)  # copy
    if isinstance(named_axis, tuple):
        named_axis_mapping = _axis_tuple_to_mapping(named_axis)
    elif isinstance(named_axis, dict):
        _check_axis_mapping_unique_values(named_axis)
        named_axis_mapping = {**attrs.get(_NamedAxisKey, {}), **named_axis}
    else:
        raise TypeError(f"named_axis must be a tuple or dict, not {named_axis}")

    if _NamedAxisKey in attrs and not overwrite:
        raise KeyError(
            f"Can't set named axis mapping into attrs with key {_NamedAxisKey}, have {attrs=}."
        )

    attrs[_NamedAxisKey] = named_axis_mapping
    return attrs


# These are the strategies to handle named axis for the
# output array when performing operations along an axis.
# See studies/named_axis.md#named-axis-in-high-level-functions and
# https://pytorch.org/docs/stable/name_inference.html.
#
# The strategies are:
# - "keep all" (_identity_named_axis): Keep all named axes in the output array, e.g.: `ak.drop_none`
# - "keep one" (_keep_named_axis): Keep one named axes in the output array, e.g.: `ak.firsts`
# - "remove all" (_remove_all_named_axis): Removes all named axis, e.g.: `ak.categories
# - "remove one" (_remove_named_axis): Remove the named axis from the output array, e.g.: `ak.sum`
# - "unify" (_unify_named_axis): Unify the named axis in the output array given two input arrays, e.g.: `__add__`
# - "collapse" (_collapse_named_axis): Collapse multiple named axis to None in the output array, e.g.: `ak.flatten`
# - "permute" (_permute_named_axis): Permute the named axis in the output array, e.g.: `ak.transpose` (does this exist?)
# - "contract" (_contract_named_axis): Contract the named axis in the output array, e.g.: `matmul` (does this exist?)
# - "adding" (_adding_named_axis): Add a new named axis to the output array, e.g.: `ak.concatenate` (not clear yet...)


def _identity_named_axis(
    named_axis: AxisTuple,
) -> AxisTuple:
    """
    Determines the new named axis after keeping all axes. This is useful, for example,
    when applying an operation that does not change the axis structure.

    Args:
        named_axis (AxisTuple): The current named axis.

    Returns:
        AxisTuple: The new named axis after keeping all axes.

    Examples:
        >>> _identity_named_axis(("x", "y", "z"))
        ("x", "y", "z")
    """
    return tuple(named_axis)


def _keep_named_axis(
    named_axis: AxisTuple,
    axis: int | None = None,
) -> AxisTuple:
    """
    Determines the new named axis after keeping the specified axis. This is useful, for example,
    when applying an operation that keeps only one axis.

    Args:
        named_axis (AxisTuple): The current named axis.
        axis (int | None, optional): The index of the axis to keep. If None, all axes are kept. Default is None.

    Returns:
        AxisTuple: The new named axis after keeping the specified axis.

    Examples:
        >>> _keep_named_axis(("x", "y", "z"), 1)
        ("y",)
        >>> _keep_named_axis(("x", "y", "z"))
        ("x", "y", "z")
    """
    return tuple(named_axis) if axis is None else (named_axis[axis],)


def _remove_all_named_axis(
    named_axis: AxisTuple,
    n: int | None = None,
) -> AxisTuple:
    """
    Determines the new named axis after removing all axes. This is useful, for example,
    when applying an operation that removes all axes.

    Args:
        named_axis (AxisTuple): The current named axis.
        n (int | None, optional): The number of axes to remove. If None, all axes are removed. Default is None.

    Returns:
        AxisTuple: The new named axis after removing all axes. All elements will be None.

    Examples:
        >>> _remove_all_named_axis(("x", "y", "z"))
        (None, None, None)
        >>> _remove_all_named_axis(("x", "y", "z"), 2)
        (None, None)
    """
    return (None,) * (len(named_axis) if n is None else n)


def _remove_named_axis(
    axis: int | None,
    named_axis: AxisTuple,
) -> AxisTuple:
    """
    Determines the new named axis after removing the specified axis. This is useful, for example,
    when applying a sum operation along an axis.

    Args:
        axis (int): The index of the axis to remove.
        named_axis (AxisTuple): The current named axis.

    Returns:
        AxisTuple: The new named axis after removing the specified axis.

    Examples:
        >>> _remove_named_axis(1, ("x", "y", "z"))
        ("x", "z")
    """
    if axis is None:
        return (None,)
    return tuple(name for i, name in enumerate(named_axis) if i != axis)


def _permute_named_axis(
    axis: int,
    named_axis: AxisTuple,
) -> AxisTuple:
    raise NotImplementedError()


def _unify_named_axis(
    named_axis1: AxisTuple,
    named_axis2: AxisTuple,
) -> AxisTuple:
    """
    Unifies two named axes into a single named axis. If the axes are identical or if one of them is None,
    the unified axis will be the non-None axis. If the axes are different and neither of them is None,
    a ValueError is raised.

    Args:
        named_axis1 (AxisTuple): The first named axis to unify.
        named_axis2 (AxisTuple): The second named axis to unify.

    Returns:
        AxisTuple: The unified named axis.

    Raises:
        ValueError: If the axes are different and neither of them is None.

    Examples:
        >>> _unify_named_axis(("x", "y", None), ("x", "y", "z"))
        ("x", "y", "z")
        >>> _unify_named_axis(("x", "y", "z"), ("x", "y", "z"))
        ("x", "y", "z")
        >>> _unify_named_axis(("x", "y", "z"), (None, None, None))
        ("x", "y", "z")
        >>> _unify_named_axis(("x", "y", "z"), ("a", "b", "c"))
        ValueError: Cannot unify different axes: 'x' and 'a'
    """
    result = []
    for ax1, ax2 in zip(named_axis1, named_axis2):
        if ax1 == ax2 or ax1 is None or ax2 is None:
            result.append(ax1 if ax1 is not None else ax2)
        else:
            raise ValueError(f"Cannot unify different axes: '{ax1}' and '{ax2}'")
    return tuple(result)


def _collapse_named_axis(
    axis: tuple[int, ...] | int | None,
    named_axis: AxisTuple,
) -> AxisTuple:
    """
    Determines the new named axis after collapsing the specified axis. This is useful, for example,
    when applying a flatten operation along an axis.

    Args:
        axis (tuple[int, ...] | int | None): The index of the axis to collapse. If None, all axes are collapsed.
        named_axis (AxisTuple): The current named axis.

    Returns:
        AxisTuple: The new named axis after collapsing the specified axis.

    Examples:
        >>> _collapse_named_axis(1, ("x", "y", "z"))
        ("x", "z")
        >>> _collapse_named_axis(None, ("x", "y", "z"))
        (None,)
        >>> _collapse_named_axis((1, 2), ("x", "y", "z"))
        ("x",)
        >>> _collapse_named_axis((0, 1, 2), ("x", "y", "z"))
        (None,)
        >>> _collapse_named_axis((0, 2), ("x", "y", "z"))
        ("y",)
    """
    if axis is None:
        return (None,)
    elif isinstance(axis, int):
        axis = (axis,)
    return tuple(name for i, name in enumerate(named_axis) if i not in axis) or (None,)


class Slicer:
    """
    Provides a more convenient syntax for slicing.

    Examples:
        Create a Slicer object:

        >>> ak_slice = Slicer()

        Use the Slicer object to create slices:

        >>> ak_slice[1:5]
        slice(1, 5)

        >>> ak_slice[1:5:2]
        slice(1, 5, 2)

        Create a tuple of slices:

        >>> ak_slice[1:5:2, 2:10]
        (slice(1, 5, 2), slice(2, 10))

        Use the Slicer object to create a slice that includes all elements:

        >>> ak_slice[...]
        slice(None)

        >>> ak_slice[:]
        slice(None)
    """

    def __getitem__(self, where):
        return where
