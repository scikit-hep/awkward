from __future__ import annotations

import awkward._typing as tp
from awkward._regularize import is_integer

# axis names are hashables, mostly strings,
# except for integers, which are reserved for positional axis.
AxisName: tp.TypeAlias = tp.Hashable

# e.g.: {"x": 0, "y": 1, "z": 2}
AxisMapping: tp.TypeAlias = tp.Mapping[AxisName, int]

# e.g.: ("x", "y", None) where None is a wildcard
AxisTuple: tp.TypeAlias = tp.Tuple[AxisName, ...]


_NamedAxisKey: str = "__named_axis__"  # reserved for named axis


class AttrsNamedAxisMapping(tp.TypedDict, total=False):
    _NamedAxisKey: AxisMapping


@tp.runtime_checkable
class MaybeSupportsNamedAxis(tp.Protocol):
    @property
    def attrs(self) -> tp.Mapping | AttrsNamedAxisMapping: ...


# just a class for inplace mutation
class NamedAxis:
    mapping: AxisMapping


NamedAxis.mapping = {}


def _get_named_axis(
    ctx: MaybeSupportsNamedAxis | AttrsNamedAxisMapping | tp.Mapping,
) -> AxisMapping:
    """
    Retrieves the named axis from the provided context. The context can either be an object that supports named axis
    or a dictionary that includes a named axis mapping.

    Args:
        ctx (MaybeSupportsNamedAxis | AttrsNamedAxisMapping): The context from which the named axis is to be retrieved.

    Returns:
        AxisMapping: The named axis retrieved from the context. If the context does not include a named axis,
            an empty dictionary is returned.

    Examples:
        >>> class Test(MaybeSupportsNamedAxis):
        ...     @property
        ...     def attrs(self):
        ...         return {_NamedAxisKey: {"x": 0, "y": 1, "z": 2}}
        ...
        >>> _get_named_axis(Test())
        {"x": 0, "y": 1, "z": 2}
        >>> _get_named_axis({_NamedAxisKey: {"x": 0, "y": 1, "z": 2}})
        {"x": 0, "y": 1, "z": 2}
        >>> _get_named_axis({"other_key": "other_value"})
        {}
    """
    if isinstance(ctx, MaybeSupportsNamedAxis):
        return _get_named_axis(ctx.attrs)
    elif isinstance(ctx, tp.Mapping) and _NamedAxisKey in ctx:
        return dict(ctx[_NamedAxisKey])
    else:
        return {}


def _supports_named_axis(ctx: MaybeSupportsNamedAxis | AttrsNamedAxisMapping) -> bool:
    """Check if the given ctx supports named axis.

    Args:
        ctx (SupportsNamedAxis or AttrsNamedAxisMapping): The ctx to check.

    Returns:
        bool: True if the ctx supports named axis, False otherwise.
    """
    return bool(_get_named_axis(ctx))


def _make_positional_axis_tuple(n: int) -> tuple[int, ...]:
    """
    Generates a positional axis tuple of length n.

    Args:
        n (int): The length of the positional axis tuple to generate.

    Returns:
        tuple[int, ...]: The generated positional axis tuple.

    Examples:
        >>> _make_positional_axis_tuple(3)
        (0, 1, 2)
    """
    return tuple(range(n))


def _is_valid_named_axis(axis: AxisName) -> bool:
    """
    Checks if the given axis is a valid named axis. A valid named axis is a hashable object that is not an integer.

    Args:
        axis (AxisName): The axis to check.

    Returns:
        bool: True if the axis is a valid named axis, False otherwise.

    Examples:
        >>> _is_valid_named_axis("x")
        True
        >>> _is_valid_named_axis(NamedAxisMarker())
        True
        >>> _is_valid_named_axis(1)
        False
    """
    return (
        # axis must be hashable
        isinstance(axis, AxisName)
        # ... but not an integer, otherwise we would confuse it with positional axis
        and not is_integer(axis)
        # Let's only allow strings for now, in the future we can open up to more types
        # by removing the isinstance(axis, str) check.
        and isinstance(axis, str)
    )


def _check_valid_axis(axis: AxisName) -> AxisName:
    """
    Checks if the given axis is a valid named axis. If not, raises a ValueError.

    Args:
        axis (AxisName): The axis to check.

    Returns:
        AxisName: The axis if it is a valid named axis.

    Raises:
        ValueError: If the axis is not a valid named axis.

    Examples:
        >>> _check_valid_axis("x")
        "x"
        >>> _check_valid_axis(1)
        Traceback (most recent call last):
        ...
        ValueError: Axis names must be hashable and not int, got 1
    """
    if not _is_valid_named_axis(axis):
        raise ValueError(f"Axis names must be hashable and not int, got {axis!r}")
    return axis


def _axis_tuple_to_mapping(axis_tuple: AxisTuple) -> AxisMapping:
    """
    Converts a tuple of axis names to a dictionary mapping axis names to their positions.

    Args:
        axis_tuple (AxisTuple): A tuple of axis names. Can include None as a wildcard.

    Returns:
        AxisMapping: A dictionary mapping axis names to their positions.

    Examples:
        >>> _axis_tuple_to_mapping(("x", None, "y"))
        {"x": 0, "y": 2}
    """
    return {axis: i for i, axis in enumerate(axis_tuple) if axis is not None}


def _named_axis_to_positional_axis(
    named_axis: AxisMapping,
    axis: AxisName,
) -> int:
    """
    Converts a single named axis to a positional axis.

    Args:
        axis (AxisName): The named axis to convert.
        named_axis (AxisMapping): The mapping from named axes to positional axes.

    Returns:
        int | None: The positional axis corresponding to the given named axis. If the named axis is not found in the mapping, returns None.

    Raises:
        ValueError: If the named axis is not found in the named axis mapping.

    Examples:
        >>> _named_axis_to_positional_axis({"x": 0, "y": 1, "z": 2}, "x")
        0
    """
    if axis not in named_axis:
        raise ValueError(f"{axis=} not found in {named_axis=} mapping.")
    return named_axis[axis]


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
        named_axis_mapping = named_axis
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
# The possible strategies are:
# - "keep all" (_keep_named_axis(..., None)): Keep all named axes in the output array, e.g.: `ak.drop_none`
# - "keep one" (_keep_named_axis(..., int)): Keep one named axes in the output array, e.g.: `ak.firsts`
# - "keep up to" (_keep_named_axis_up_to(..., int)): Keep all named axes upto a certain positional axis in the output array, e.g.: `ak.local_index`
# - "remove all" (_remove_all_named_axis): Removes all named axis, e.g.: `ak.categories
# - "remove one" (_remove_named_axis): Remove the named axis from the output array, e.g.: `ak.sum`
# - "add one" (_add_named_axis): Add a new named axis to the output array, e.g.: `ak.concatenate, ak.singletons` (not clear yet...)
# - "unify" (_unify_named_axis): Unify the named axis in the output array given two input arrays, e.g.: `__add__`
# - "collapse" (_collapse_named_axis): Collapse multiple named axis to None in the output array, e.g.: `ak.flatten`
# - "permute" (_permute_named_axis): Permute the named axis in the output array, e.g.: `ak.transpose` (does this exist?)
# - "contract" (_contract_named_axis): Contract the named axis in the output array, e.g.: `matmul` (does this exist?)


def _keep_named_axis(
    named_axis: AxisMapping,
    axis: int | None = None,
) -> AxisMapping:
    """
    Determines the new named axis after keeping the specified axis. This function is useful when an operation
    is applied that retains only one axis.

    Args:
        named_axis (AxisMapping): The current named axis.
        axis (int | None, optional): The index of the axis to keep. If None, all axes are kept. Default is None.

    Returns:
        AxisMapping: The new named axis after keeping the specified axis.

    Examples:
        >>> _keep_named_axis({"x": 0, "y": 1, "z": 2}, 1)
        {"y": 0}
        >>> _keep_named_axis({"x": 0, "y": 1, "z": 2}, None)
        {"x": 0, "y": 1, "z": 2}
    """
    if axis is None:
        return dict(named_axis)
    return {k: 0 for k, v in named_axis.items() if v == axis}


def _keep_named_axis_up_to(
    named_axis: AxisMapping,
    axis: int,
) -> AxisMapping:
    """
    Determines the new named axis after keeping all axes up to the specified axis. This function is useful when an operation
    is applied that retains all axes up to a certain axis.

    Args:
        named_axis (AxisMapping): The current named axis.
        axis (int): The index of the axis up to which to keep.

    Returns:
        AxisMapping: The new named axis after keeping all axes up to the specified axis.

    Examples:
        >>> _keep_named_axis_up_to({"x": 0, "y": 2}, 0)
        {"x": 0}
        >>> _keep_named_axis_up_to({"x": 0, "y": 2}, 1)
        {"x": 0}
        >>> _keep_named_axis_up_to({"x": 0, "y": 2}, 2)
        {"x": 0, "y": 2}
        >>> _keep_named_axis_up_to({"x": 0, "y": -2}, 0)
        {"x": 0}
        >>> _keep_named_axis_up_to({"x": 0, "y": -2}, 1)
        {"x": 0, "y": -2}
        >>> _keep_named_axis_up_to({"x": 0, "y": -2}, 2)
        {"x": 0, "y": -2}
    """
    if axis < 0:
        raise ValueError("The axis must be a positive integer.")
    out = {}
    for k, v in named_axis.items():
        if v >= 0 and v <= axis:
            out[k] = v
        elif v < 0 and v >= -axis - 1:
            out[k] = v
    return out


def _remove_all_named_axis(
    named_axis: AxisMapping,
) -> AxisMapping:
    """
    Returns an empty named axis mapping after removing all axes from the given named axis mapping.
    This function is typically used when an operation that eliminates all axes is applied.

    Args:
        named_axis (AxisMapping): The current named axis mapping.

    Returns:
        AxisMapping: An empty named axis mapping.

    Examples:
        >>> _remove_all_named_axis({"x": 0, "y": 1, "z": 2})
        {}
    """
    return _remove_named_axis(named_axis=named_axis, axis=None)


def _remove_named_axis(
    named_axis: AxisMapping,
    axis: int | None = None,
    total: int | None = None,
) -> AxisMapping:
    """
    Determines the new named axis after removing the specified axis. This is useful, for example,
    when applying an operation that removes one axis.

    Args:
        named_axis (AxisMapping): The current named axis.
        axis (int | None, optional): The index of the axis to remove. If None, no axes are removed. Default is None.
        total (int | None, optional): The total number of axes. If None, it is calculated as the length of the named axis. Default is None.

    Returns:
        AxisMapping: The new named axis after removing the specified axis.

    Examples:
        >>> _remove_named_axis({"x": 0, "y": 1}, None)
        {}
        >>> _remove_named_axis({"x": 0, "y": 1}, 0)
        {"y": 0}
        >>> _remove_named_axis({"x": 0, "y": 1, "z": 2}, 1)
        {"x": 0, "z": 1}
        >>> _remove_named_axis({"x": 0, "y": 1, "z": -1}, 1)
        {"x": 0, "z": -1}
        >>> _remove_named_axis({"x": 0, "y": 1, "z": -3}, 1)
        {"x": 0, "z": -2}
    """
    if axis is None:
        return {}

    if total is None:
        total = len(named_axis)

    # remove the specified axis
    out = {ax: pos for ax, pos in named_axis.items() if pos != axis}

    return _adjust_pos_axis(out, axis, total)


def _remove_named_axis_by_name(
    named_axis: AxisMapping,
    axis: AxisName,
    total: int | None = None,
) -> AxisMapping:
    """
    Determines the new named axis after removing the specified axis by its name. This is useful, for example,
    when applying an operation that removes one axis.

    Args:
        named_axis (AxisMapping): The current named axis.
        axis (AxisName | None, optional): The name of the axis to remove. If None, no axes are removed. Default is None.
        total (int | None, optional): The total number of axes. If None, it is calculated as the length of the named axis. Default is None.

    Returns:
        AxisMapping: The new named axis after removing the specified axis.

    Examples:
        >>> _remove_named_axis_by_name({"x": 0, "y": 1}, "x")
        {"y": 0}
        >>> _remove_named_axis_by_name({"x": 0, "y": 1, "z": 2}, "y")
        {"x": 0, "z": 1}
        >>> _remove_named_axis_by_name({"x": 0, "y": 1, "z": -1}, "z")
        {"x": 0, "y": 1}
    """
    if axis is None:
        return {}

    if total is None:
        total = len(named_axis)

    # remove the specified axis
    out = dict(named_axis)
    pos = out.pop(axis)

    return _adjust_pos_axis(out, pos, total)


def _adjust_pos_axis(
    named_axis: AxisMapping,
    axis: int,
    total: int,
) -> AxisMapping:
    """
    Adjusts the positions of the axes in the named axis mapping after an axis has been removed.

    The adjustment is done as follows:
    - If the position of an axis is greater than the removed axis, it is decremented by 1.
    - If the position of an axis is less than the removed axis and greater or equal to -1, it is kept as is.
    - If the position of an axis is negative and smaller than the amount of total left axes, it is incremented by 1.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        axis (int): The position of the removed axis.
        total (int): The total number of axes.

    Returns:
        AxisMapping: The adjusted named axis mapping.

    Examples:
        >>> _adjust_pos_axis({"x": 0, "z": 2}, 1, 3)
        {"x": 0, "z": 1}
        >>> _adjust_pos_axis({"x": 0, "z": -1}, 1, 3)
        {"x": 0, "z": -1}
        >>> _adjust_pos_axis({"x": 0, "z": -3}, 1, 3)
        {"x": 0, "z": -2}
    """
    out = dict(named_axis)
    for k, v in out.items():
        if v > axis:
            out[k] = v - 1
        elif v < -1 and len(out) < total:
            out[k] = v + 1
        else:
            out[k] = v
    return out


def _add_named_axis(
    named_axis: AxisMapping,
    axis: int,
) -> AxisMapping:
    """
    Adds a new axis to the named_axis at the specified position.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        axis (int): The position at which to add the new axis.

    Returns:
        AxisMapping: The updated named axis mapping after adding the new axis.

    Examples:
        >>> _add_named_axis({"x": 0, "y": 1, "z": 2}, 0)
        {"x": 1, "y": 2, "z": 3}
        >>> _add_named_axis({"x": 0, "y": 1, "z": 2}, 1)
        {"x": 0, "y": 2, "z": 3}
    """
    out = dict(named_axis)
    for k, v in out.items():
        if v >= axis and v >= 0:
            out[k] = v + 1
    return out


def _permute_named_axis(
    axis: int,
    named_axis: AxisTuple,
) -> AxisTuple:
    raise NotImplementedError()


def _unify_named_axis(
    named_axis1: AxisMapping,
    named_axis2: AxisMapping,
) -> AxisMapping:
    """
    Unifies two named axes into a single named axis. The function iterates over all positional axes present in either of the input named axes.
    For each positional axis, it checks the corresponding axis names in both input named axes. If the axis names are the same or if one of them is None,
    the unified axis will be the non-None axis. If the axis names are different and neither of them is None, a ValueError is raised.

    Args:
        named_axis1 (AxisMapping): The first named axis to unify.
        named_axis2 (AxisMapping): The second named axis to unify.

    Returns:
        AxisMapping: The unified named axis.

    Raises:
        ValueError: If the axes are different and neither of them is None.

    Examples:
        >>> _unify_named_axis({"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1, "z": 2})
        {"x": 0, "y": 1, "z": 2}

        >>> _unify_named_axis({"x": 0, "y": 1}, {"x": 0, "y": 1, "z": 2})
        {"x": 0, "y": 1, "z": 2}

        >>> _unify_named_axis({"x": 0, "y": 1, "z": 2}, {"a": 0, "b": 1, "c": 2})
        Traceback (most recent call last):
        ...
        ValueError: The named axes are different. Got: 'x' and 'a' for positional axis 0

        >>> _unify_named_axis({"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1, "z": 3})
        {"x": 0, "y": 1, "z": 2}

        >>> _unify_named_axis({"x": 0, "y": 1, "z": 2}, {})
        {"x": 0, "y": 1, "z": 2}

        >>> _unify_named_axis({}, {"x": 0, "y": 1, "z": 2})
        {"x": 0, "y": 1, "z": 2}

        >>> _unify_named_axis({}, {})
        {}
    """

    def _get_axis_name(
        axis_mapping: AxisMapping, positional_axis: int
    ) -> AxisName | None:
        for name, position in axis_mapping.items():
            if position == positional_axis:
                return name
        return None

    unified_named_axis = {}
    all_positional_axes = set(named_axis1.values()) | set(named_axis2.values())
    for position in all_positional_axes:
        axis_name1 = _get_axis_name(named_axis1, position)
        axis_name2 = _get_axis_name(named_axis2, position)
        if axis_name1 is not None and axis_name2 is not None:
            if axis_name1 != axis_name2:
                raise ValueError(
                    f"The named axes are incompatible. Got: {axis_name1} and {axis_name2} for positional axis {position}"
                )
            unified_named_axis[axis_name1] = position
        elif axis_name1 is not None:  # axis_name2 is None
            unified_named_axis[axis_name1] = position
        elif axis_name2 is not None:  # axis_name1 is None
            unified_named_axis[axis_name2] = position
    return unified_named_axis


def _collapse_named_axis(
    named_axis: AxisMapping,
    axis: tuple[int, ...] | int | None,
) -> AxisMapping:
    """
    Determines the new named axis after collapsing the specified axis. This is useful, for example,
    when applying a flatten operation along an axis.

    Args:
        axis (tuple[int, ...] | int | None): The index of the axis to collapse. If None, all axes are collapsed.
        named_axis (AxisMapping): The current named axis.

    Returns:
        AxisMapping: The new named axis after collapsing the specified axis.
    """
    raise NotImplementedError()


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


# Define a type alias for a slice or int (can be a single axis or a sequence of axes)
AxisSlice: tp.TypeAlias = tp.Union[tuple, slice, int, tp.EllipsisType, None]
NamedAxisSlice: tp.TypeAlias = tp.Dict[AxisName, AxisSlice]


def _normalize_named_slice(
    named_axis: AxisMapping,
    where: AxisSlice | NamedAxisSlice | tuple[AxisSlice | NamedAxisSlice],
    total: int,
) -> AxisSlice:
    """
    Normalizes a named slice into a positional slice.

    This function takes a named slice (a dictionary mapping axis names to slices) and converts it into a positional slice
    (a tuple of slices). The positional slice can then be used to index an array.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        where (AxisSlice | NamedAxisSlice | tuple[AxisSlice | NamedAxisSlice]): The slice to normalize. Can be a single slice, a tuple of slices, or a dictionary mapping axis names to slices.
        total (int): The total number of axes.

    Returns:
        AxisSlice: The normalized slice.

    Raises:
        ValueError: If an invalid axis name is provided in the slice.

    Examples:
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {0: 0}, 3)
        (0, slice(None), slice(None))
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {-1: 0}, 3)
        (slice(None), slice(None), 0)
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": 0}, 3)
        (0, slice(None), slice(None))
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1}, 3)
        (0, 1, slice(None))
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1, "z": ...}, 3)
        (0, 1, ...)
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1, "z": slice(0, 1)}, 3)
        (0, 1, slice(0, 1))
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": (0, 1)}, 3)
        ((0, 1), slice(None), slice(None))
        >>> _normalize_named_slice({"x": 0, "y": 1, "z": 2}, {"x": [0, 1]}, 3)
        ([0, 1], slice(None), slice(None))
    """
    if isinstance(where, dict):
        out_where: list[AxisSlice] = [slice(None)] * total
        for ax_name, ax_where in where.items():
            slice_ = ax_where if ax_where is not ... else slice(None)
            if isinstance(ax_name, int):
                out_where[ax_name] = slice_
            elif _is_valid_named_axis(ax_name):
                idx = _named_axis_to_positional_axis(named_axis, ax_name)
                out_where[idx] = slice_
            else:
                raise ValueError(f"Invalid axis name: {ax_name} in slice {where}")
        where = tuple(out_where)
    return where
