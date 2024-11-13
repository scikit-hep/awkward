from __future__ import annotations

import json
import re
from dataclasses import dataclass

import awkward._typing as tp
from awkward._layout import _neg2pos_axis
from awkward._regularize import is_integer

# axis names are hashables, mostly strings,
# except for integers, which are reserved for positional axis.
AxisName: tp.TypeAlias = tp.Hashable

# e.g.: {"x": 0, "y": 1, "z": 2}
AxisMapping: tp.TypeAlias = tp.Mapping[AxisName, int]

# e.g.: ("x", "y", None) where None is a wildcard
AxisTuple: tp.TypeAlias = tp.Tuple[AxisName, ...]


NAMED_AXIS_KEY: tp.Literal["__named_axis__"] = (
    "__named_axis__"  # reserved for named axis
)


# just a class for inplace mutation
class NamedAxis:
    mapping: AxisMapping


NamedAxis.mapping = {}


def _prettify_named_axes(
    named_axis: AxisMapping,
    delimiter: str = ", ",
    maxlen: None | int = None,
) -> str:
    """
    This function takes a named axis mapping and returns a string representation of the mapping.
    The axis names are sorted in ascending order of their corresponding integer values.
    If the axis name is a valid Python identifier, it is represented as is.
    Otherwise, it is represented as a JSON string.

    Args:
        named_axis (AxisMapping): The named axis mapping to prettify.
        delimiter (str, optional): The delimiter to use between items in the output string. Defaults to ", ".
        maxlen (None | int, optional): The maximum length of the output string. If the string exceeds this length, it is truncated and ends with "...". Defaults to None.

    Returns:
        str: The prettified string representation of the named axis mapping.

    Examples:
        >>> _prettify_named_axes({"x": 0, "y": 1, "z": 2})
        'x:0, y:1, z:2'
        >>> _prettify_named_axes({"x": 0, "y": 1, "$": 2})
        'x:0, y:1, "$":2'
        >>> _prettify_named_axes({"x": 0, "y": 1, "z": 2}, delimiter="; ")
        'x:0; y:1; z:2'
        >>> _prettify_named_axes({"foo": 0, "bar": 1, "baz": 2}, maxlen=17)
        'foo:0, bar:1, ...'
    """

    def _prettify(ax: AxisName) -> str:
        repr_ax = str(ax)
        if re.match("[A-Za-z_][A-Za-z_0-9]*", repr_ax):
            return repr_ax
        return json.dumps(repr_ax)

    sorted_named_axis = sorted(named_axis.items(), key=lambda x: x[1])
    items = [
        f"{_prettify(named_ax)}:{pos_ax}" for named_ax, pos_ax in sorted_named_axis
    ]
    if maxlen is not None:
        if len(delimiter.join(items)) > maxlen:
            while (
                len(delimiter.join(items)) > maxlen - len(delimiter + "...")
            ) and items:
                items.pop(-1)
            items.append("...")
    return delimiter.join(items)


def _get_named_axis(ctx: tp.Any, allow_any: bool = False) -> AxisMapping:
    """
    Retrieves the named axis from the provided context.

    Args:
        ctx (Any): The context from which the named axis is to be retrieved.

    Returns:
        AxisMapping: The named axis retrieved from the context. If the context does not include a named axis,
            an empty dictionary is returned.

    Examples:
        >>> _get_named_axis(ak.Array([1, 2, 3], named_axis={"x": 0}))
        {"x": 0}
        >>> _get_named_axis(np.array([1, 2, 3]))
        {}
        >>> _get_named_axis({NAMED_AXIS_KEY: {"x": 0, "y": 1, "z": 2}})
        {"x": 0, "y": 1, "z": 2}
        >>> _get_named_axis({"other_key": "other_value"})
        {}
    """
    from awkward._layout import HighLevelContext
    from awkward.highlevel import Array, Record

    if hasattr(ctx, "attrs") and (
        isinstance(ctx, (HighLevelContext, Array, Record)) or allow_any
    ):
        return _get_named_axis(ctx.attrs, allow_any=True)
    elif allow_any and isinstance(ctx, tp.Mapping) and NAMED_AXIS_KEY in ctx:
        return dict(ctx[NAMED_AXIS_KEY])
    else:
        return {}


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
    Checks if the given axis is a valid named axis. A valid named axis is a hashable object that is not an integer or None. Currently it is restricted to strings.

    Args:
        axis (AxisName): The axis to check.

    Returns:
        bool: True if the axis is a valid named axis, False otherwise.

    Examples:
        >>> _is_valid_named_axis("x")
        True
        >>> _is_valid_named_axis(1)
        False
    """
    return (
        # axis must be hashable
        isinstance(axis, AxisName)
        # ... but not an integer, otherwise we would confuse it with positional axis
        and not is_integer(axis)
        # we also prohibit None, which is reserved for wildcard
        and axis is not None
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
        ValueError: Axis names must be hashable and not int, got 1 [type(axis)=<class 'int'>]
    """
    if not _is_valid_named_axis(axis):
        raise ValueError(
            f"Axis names must be hashable and not int, got {axis!r} [{type(axis)=}]"
        )
    return axis


def _check_valid_named_axis_mapping(named_axis: AxisMapping) -> AxisMapping:
    """
    Checks if the given named axis mapping is valid. A valid named axis mapping is a dictionary where the keys are valid named axes
    (hashable objects that are not integers) and the values are integers.

    Args:
        named_axis (AxisMapping): The named axis mapping to check.

    Raises:
        ValueError: If any of the keys in the named axis mapping is not a valid named axis or if any of the values is not an integer.

    Examples:
        >>> _check_valid_named_axis_mapping({"x": 0, "y": 1, "z": 2})  # No exception is raised
        >>> _check_valid_named_axis_mapping({"x": 0, "y": 1, "z": "2"})
        Traceback (most recent call last):
        ...
        ValueError: Named axis mapping values must be integers, got '2' [type(axis)=<class 'str'>]
        >>> _check_valid_named_axis_mapping({"x": 0, 1: 1, "z": 2})
        Traceback (most recent call last):
        ...
        ValueError: Axis names must be hashable and not int, got 1 [type(axis)=<class 'int'>]
    """
    for name, axis in named_axis.items():
        _check_valid_axis(name)
        if not is_integer(axis):
            raise ValueError(
                f"Named axis mapping values must be integers, got {axis!r} [{type(axis)=}]"
            )
    return named_axis


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


def _prepare_named_axis_for_attrs(
    named_axis: AxisMapping | AxisTuple,
    ndim: int,
) -> AxisMapping:
    """
    Prepares the named axis for attribute assignment.

    This function takes a named axis, which can either be a mapping or a tuple, and returns a dictionary mapping axis names to their positions.
    The function checks if the named axis is valid and if the positional axes match the number of dimensions. If not, an error is raised.

    Args:
        named_axis (AxisMapping | AxisTuple): The named axis to prepare. Can either be a mapping or a tuple.
        ndim (int): The number of dimensions.

    Returns:
        AxisMapping: The prepared named axis.

    Raises:
        TypeError: If the named axis is not a mapping or a tuple.
        ValueError: If the named axes do not point to positional axes matching the number of dimensions.

    Examples:
        >>> _prepare_named_axis_for_attrs({"x": 0, "y": 1, "z": 2}, 3)
        {"x": 0, "y": 1, "z": 2}
        >>> _prepare_named_axis_for_attrs(("x", "y", "z"), 3)
        {"x": 0, "y": 1, "z": 2}
        >>> _prepare_named_axis_for_attrs({"x": 0, "y": 1, "z": 2}, 2)
        Traceback (most recent call last):
        ...
        ValueError: Named axes must point to positional axes matching 2 dimensions, got named_axis={"x": 0, "y": 1, "z": 2}, ndim=2
    """
    if isinstance(named_axis, tuple):
        _named_axis = _axis_tuple_to_mapping(named_axis)
    elif isinstance(named_axis, dict):
        _named_axis = named_axis
    else:
        raise TypeError(
            f"named_axis must be a mapping or a tuple, got {named_axis=} [{type(named_axis)=}]"
        )
    _check_valid_named_axis_mapping(_named_axis)
    pos_axes = set(_named_axis.values())
    if max(pos_axes, default=0) >= ndim or min(pos_axes, default=0) < -ndim:
        raise ValueError(
            f"Named axes must point to positional axes matching {ndim} dimensions, got {named_axis=}, {ndim=}"
        )
    return _named_axis


def _make_named_int_class(name: tp.Any) -> type[int]:
    class NamedInt(int):
        def __repr__(self):
            value_repr = super().__repr__()
            return f"{name!r} (named axis) -> {value_repr} (pos. axis)"

        __str__ = __repr__

    return NamedInt


def _named_axis_to_positional_axis(
    named_axis: AxisMapping,
    axis: AxisName,
) -> int | None:
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
    if _is_valid_named_axis(axis):
        if axis not in named_axis:
            raise ValueError(f"{axis=} not found in {named_axis=} mapping.")

        # we wrap it to preserve the original name in its __repr__ and __str__
        # in order to properly display it in error messages. This is useful for cases
        # where the positional axis is pointing to a non-existing axis. The error message
        # will then show the original (named) axis together with the positional axis.
        cls = _make_named_int_class(axis)
        return cls(named_axis[axis])

    if is_integer(axis):
        # TODO: is_integer is an external  helper function that doesn't specify types
        return int(tp.cast(tp.Any, axis))
    elif axis is None:
        return None
    else:
        raise ValueError(f"Invalid {axis=} [{type(axis)=}]")


# These are the strategies to handle named axis for the
# output array when performing operations along an axis.
# See studies/named_axis.md#named-axis-in-high-level-functions and
# https://pytorch.org/docs/stable/name_inference.html.
#
# The possible strategies are:
# - "keep all" (_keep_named_axis(..., None)): Keep all named axes in the output array, e.g.: `ak.drop_none`
# - "keep one" (_keep_named_axis(..., int)): Keep one named axes in the output array, e.g.: `ak.firsts`
# - "keep up to" (_keep_named_axis_up_to(..., int)): Keep all named axes up to a certain positional axis in the output array, e.g.: `ak.local_index`
# - "remove all" (_remove_all_named_axis): Removes all named axis, e.g.: `ak.categories`
# - "remove one" (_remove_named_axis): Remove the named axis from the output array, e.g.: `ak.sum`
# - "add one" (_add_named_axis): Add a new named axis to the output array, e.g.: `ak.concatenate`
# - "unify" (_unify_named_axis): Unify the named axis in the output array given two input arrays, e.g.: `ak.broadcast_arrays`


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
        return named_axis
    return {k: 0 for k, v in named_axis.items() if v == axis}


def _keep_named_axis_up_to(
    named_axis: AxisMapping,
    axis: int,
    total: int,
) -> AxisMapping:
    """
    Determines the new named axis after keeping all axes up to the specified axis. This function is useful when an operation
    is applied that retains all axes up to a certain axis.

    Args:
        named_axis (AxisMapping): The current named axis.
        axis (int): The index of the axis up to which to keep.
        total (int): The total number of axes.

    Returns:
        AxisMapping: The new named axis after keeping all axes up to the specified axis.

    Examples:
        >>> _keep_named_axis_up_to({"x": 0, "y": 1, "z": 2}, 1, 3)
        {"x": 0, "y": 1}
        >>> _keep_named_axis_up_to({"x": 0, "y": 1, "z": 2}, -1, 3)
        {"x": 0, "y": 1, "z": 2}
        >>> _keep_named_axis_up_to({"x": 0, "y": 1, "z": 2}, 0, 3)
        {"x": 0}
    """
    axis = _neg2pos_axis(axis, total)
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
    out = {
        ax: pos
        for ax, pos in named_axis.items()
        if _neg2pos_axis(pos, total) != _neg2pos_axis(axis, total)
    }

    return _adjust_pos_axis(out, axis, total, direction=-1)


def _adjust_pos_axis(
    named_axis: AxisMapping,
    axis: int,
    total: int,
    direction: int,
) -> AxisMapping:
    """
    Adjusts the positions of the axes in the named axis mapping after an axis has been removed or added.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        axis (int): The position of the removed/added axis.
        total (int): The total number of axes.
        direction (int): The direction of the adjustment. -1 means axis is removed; +1 means axis is added. Default is +1.

    Returns:
        AxisMapping: The adjusted named axis mapping.

    Examples:
        # axis=1 removed
        >>> _adjust_pos_axis({"x": 0, "z": 2}, 1, 3, -1)
        {"x": 0, "z": 1}
        # axis=1 added
        >>> _adjust_pos_axis({"x": 0, "z": 2}, 1, 3, +1)
        {"x": 0, "z": 3}
        # axis=1 removed
        >>> _adjust_pos_axis({"x": 0, "z": -1}, 1, 3, -1)
        {"x": 0, "z": -1}
        # axis=1 added
        >>> _adjust_pos_axis({"x": 0, "z": -1}, 1, 3, +1)
        {"x": 0, "z": -1}
    """
    assert direction in (-1, +1), f"Invalid direction: {direction}"

    def _adjust(pos: int, axis: int, direction: int) -> int:
        # positive axis
        if axis >= 0:
            # positive axis and position greater than or equal to the removed/added (positive) axis
            # -> change position by direction
            if pos >= axis:
                return pos + direction
            # positive axis and negative position
            # -> change position by direction
            elif pos < 0 and pos + total < axis:
                return pos - direction
            # positive axis and position smaller than the removed/added (positive) axis, but greater than 0
            # -> keep position
            else:
                return pos
        # negative axis
        else:
            # negative axis and position smaller than the removed/added (negative) axis
            # -> change position by inverse direction
            if pos <= axis:
                return pos - direction
            # negative axis and positive position
            # -> change position by inverse direction
            elif pos > 0 and pos > axis + total:
                return pos + direction
            # negative axis and position greater than the removed/added (negative) axis, but smaller than 0
            # -> keep position
            else:
                return pos

    return {k: _adjust(v, axis, direction) for k, v in named_axis.items()}


def _add_named_axis(
    named_axis: AxisMapping,
    axis: int,
    total: int | None = None,
) -> AxisMapping:
    """
    Adds a new axis to the named_axis at the specified position.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        axis (int): The position at which to add the new axis.
        total (int | None): The total number of axes.

    Returns:
        AxisMapping: The updated named axis mapping after adding the new axis.

    Examples:
        >>> _add_named_axis({"x": 0, "y": 1, "z": 2}, 0)
        {"x": 1, "y": 2, "z": 3}
        >>> _add_named_axis({"x": 0, "y": 1, "z": 2}, 1)
        {"x": 0, "y": 2, "z": 3}
    """
    if total is None:
        total = len(named_axis)

    return _adjust_pos_axis(named_axis, axis, total, direction=+1)


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


@dataclass
class NamedAxesWithDims:
    """
    A dataclass that stores the named axis and their corresponding dimensions.
    This is a helper class to store the named axis mapping and the number of
    dimensions of each named axis, which is useful for broadcasting.

    Attributes:
        named_axis (AxisMapping): The named axis mapping.
        ndims (Tuple[int]): The number of dimensions of the named axis.
    """

    named_axis: list[AxisMapping]
    ndims: list[int | None]

    def __post_init__(self):
        if len(self.named_axis) != len(self.ndims):
            raise ValueError(
                "The number of dimensions must match the number of named axis mappings."
            )

    def __iter__(self) -> tp.Iterator[tuple[AxisMapping, int | None]]:
        yield from zip(self.named_axis, self.ndims)

    @classmethod
    def prepare_contexts(
        cls, arrays: tp.Sequence, unwrap_kwargs: dict | None = None
    ) -> tuple[dict, dict]:
        from awkward._layout import HighLevelContext
        from awkward._typetracer import MaybeNone

        # unwrap options
        arrays = [x.content if isinstance(x, MaybeNone) else x for x in arrays]

        _unwrap_kwargs = {"allow_unknown": True}
        if unwrap_kwargs is not None:
            _unwrap_kwargs.update(unwrap_kwargs)

        _named_axes = []
        _ndims = []
        for array in arrays:
            with HighLevelContext() as ctx:
                layout = ctx.unwrap(array, **_unwrap_kwargs)
            _named_axes.append(_get_named_axis(array))
            _ndims.append(getattr(layout, "minmax_depth", (None, None))[1])

        depth_context = {NAMED_AXIS_KEY: cls(_named_axes, _ndims)}
        lateral_context = {NAMED_AXIS_KEY: cls(_named_axes, _ndims)}
        return depth_context, lateral_context

    def __setitem__(
        self, index: int, named_axis_with_ndim: tuple[AxisMapping, int | None]
    ):
        named_axis, ndim = named_axis_with_ndim
        self.named_axis[index] = named_axis
        self.ndims[index] = ndim

    def __getitem__(self, index: int) -> tuple[AxisMapping, int | None]:
        return self.named_axis[index], self.ndims[index]

    def __len__(self) -> int:
        return len(self.named_axis)


# Define a type alias for a slice or int (can be a single axis or a sequence of axes)
AxisSlice: tp.TypeAlias = tp.Union[tuple, slice, int, tp.EllipsisType, None]
NamedAxisSlice: tp.TypeAlias = tp.Dict[AxisName, AxisSlice]


def _normalize_named_slice(
    named_axis: AxisMapping,
    where: AxisSlice | NamedAxisSlice,
    total: int,
) -> AxisSlice:
    """
    Normalizes a named slice into a positional slice.

    This function takes a named slice (a dictionary mapping axis names to slices) and converts it into a positional slice
    (a tuple of slices). The positional slice can then be used to index an array.

    Args:
        named_axis (AxisMapping): The current named axis mapping.
        where (AxisSlice | NamedAxisSlice): The slice to normalize. Can be a single slice, a tuple of slices, or a dictionary mapping axis names to slices.
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
            if is_integer(ax_name):
                # it's an integer, pyright doesn't get this
                idx = tp.cast(int, ax_name)
                out_where[idx] = slice_
            elif _is_valid_named_axis(ax_name):
                # it's an integer, pyright doesn't get this
                idx = tp.cast(int, _named_axis_to_positional_axis(named_axis, ax_name))
                out_where[idx] = slice_
            else:
                raise ValueError(f"Invalid axis name: {ax_name} in slice {where}")
        where = tuple(out_where)
    return where
