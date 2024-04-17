# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from numbers import Integral

import awkward as ak
from awkward._backends.backend import Backend
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, AxisMaybeNone, Literal
from awkward.contents.content import ActionType, Content
from awkward.errors import AxisError
from awkward.forms import form
from awkward.record import Record

np = NumpyMetadata.instance()


def recursively_apply(
    layout: Content | Record,
    action: ActionType,
    behavior: dict | None = None,
    depth_context: dict[str, Any] | None = None,
    lateral_context: dict[str, Any] | None = None,
    allow_records: bool = True,
    keep_parameters: bool = True,
    numpy_to_regular: bool = True,
    return_simplified: bool = True,
    return_array: bool = True,
    function_name: str | None = None,
    regular_to_jagged=False,
) -> Content | Record | None:
    if isinstance(layout, Content):
        return layout._recursively_apply(
            action,
            1,
            copy.copy(depth_context),
            lateral_context,
            {
                "allow_records": allow_records,
                "keep_parameters": keep_parameters,
                "numpy_to_regular": numpy_to_regular,
                "regular_to_jagged": regular_to_jagged,
                "return_simplified": return_simplified,
                "return_array": return_array,
                "function_name": function_name,
            },
        )

    elif isinstance(layout, Record):
        out = recursively_apply(
            layout._array,
            action,
            behavior,
            depth_context,
            lateral_context,
            allow_records,
            keep_parameters,
            numpy_to_regular,
            return_simplified,
            return_array,
            function_name,
        )

        if return_array:
            return Record(out, layout.at)
        else:
            return None


def to_buffers(
    content: Content,
    container: MutableMapping[str, Any] | None = None,
    buffer_key="{form_key}-{attribute}",
    form_key: str | None = "node{id}",
    id_start: Integral = 0,
    backend: Backend | None = None,
    byteorder: Literal["<", ">"] = "<",
) -> tuple[form.Form, int, Mapping[str, Any]]:
    if container is None:
        container = {}
    if backend is None:
        backend = content._backend
    if not backend.nplike.known_data:
        raise TypeError("cannot call 'to_buffers' on an array without concrete data")

    if isinstance(buffer_key, str):

        def getkey(layout, form, attribute):
            return buffer_key.format(form_key=form.form_key, attribute=attribute)

    elif callable(buffer_key):

        def getkey(layout, form, attribute):
            return buffer_key(
                form_key=form.form_key,
                attribute=attribute,
                layout=layout,
                form=form,
            )

    else:
        raise TypeError(
            f"buffer_key must be a string or a callable, not {type(buffer_key)}"
        )

    if form_key is None:
        raise TypeError(
            "a 'form_key' must be supplied, to match Form elements to buffers in the 'container'"
        )

    form = content.form_with_key(form_key=form_key, id_start=id_start)

    content._to_buffers(form, getkey, container, backend, byteorder)

    return form, len(content), container


def local_index(layout: Content, axis: Integral):
    return layout._local_index(axis, 1)


def combinations(
    layout: Content,
    n: Integral,
    replacement: bool = False,
    axis: Integral = 1,
    fields: list[str] | None = None,
    parameters: dict | None = None,
):
    if n < 1:
        raise ValueError("in combinations, 'n' must be at least 1")

    recordlookup = None
    if fields is not None:
        recordlookup = fields
        if len(recordlookup) != n:
            raise ValueError("if provided, the length of 'fields' must be 'n'")
    return layout._combinations(n, replacement, recordlookup, parameters, axis, 1)


def is_unique(layout, axis: Integral | None = None) -> bool:
    negaxis = axis if axis is None else -axis
    starts = ak.index.Index64.zeros(1, nplike=layout._backend.index_nplike)
    parents = ak.index.Index64.zeros(layout.length, nplike=layout._backend.index_nplike)
    return layout._is_unique(negaxis, starts, parents, 1)


def unique(layout: Content, axis=None):
    if axis == -1 or axis is None:
        negaxis = axis if axis is None else -axis
        if negaxis is not None:
            branch, depth = layout.branch_depth
            if branch:
                if negaxis <= 0:
                    raise AxisError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves "
                        "of the tree; non-negative from the root)"
                    )
                if negaxis > depth:
                    raise AxisError(
                        f"cannot use axis={axis} on a nested list structure that splits into "
                        f"different depths, the minimum of which is depth={depth} from the leaves"
                    )
            else:
                if negaxis <= 0:
                    negaxis = negaxis + depth
                if not (0 < negaxis and negaxis <= depth):
                    raise AxisError(
                        f"axis={axis} exceeds the depth of this array ({depth})"
                    )

        starts = ak.index.Index64.zeros(1, nplike=layout._backend.index_nplike)
        parents = ak.index.Index64.zeros(
            layout.length, nplike=layout._backend.index_nplike
        )

        return layout._unique(negaxis, starts, parents, 1)

    raise AxisError(
        f"unique expects axis 'None' or '-1', got axis={axis} that is not supported yet"
    )


def pad_none(
    layout: Content, length: Integral, axis: Integral, clip: bool = False
) -> Content:
    return layout._pad_none(length, axis, 1, clip)


def remove_structure(
    layout: Content | Record,
    backend: Backend | None = None,
    flatten_records: bool = True,
    function_name: str | None = None,
    drop_nones: bool = True,
    keepdims: bool = False,
    allow_records: bool = False,
    list_to_regular: bool = False,
):
    if isinstance(layout, Record):
        return remove_structure(
            layout._array[layout._at : layout._at + 1],
            backend,
            flatten_records,
            function_name,
            drop_nones,
            keepdims,
            allow_records,
        )

    else:
        if backend is None:
            backend = layout._backend
        arrays = layout._remove_structure(
            backend,
            {
                "flatten_records": flatten_records,
                "function_name": function_name,
                "drop_nones": drop_nones,
                "keepdims": keepdims,
                "allow_records": allow_records,
                "list_to_regular": list_to_regular,
            },
        )
        return tuple(arrays)


def flatten(layout: Content, axis: int = 1) -> Content:
    offsets, flattened = layout._offsets_and_flattened(axis, 1)
    return flattened


def numbers_to_type(layout: Content, name: str, including_unknown: bool) -> Content:
    return layout._numbers_to_type(name, including_unknown)


def fill_none(layout: Content, value: Content) -> Content:
    return layout._fill_none(value)


def num(layout, axis):
    return layout._num(axis, 0)


def mergeable(one: Content, two: Content, mergebool: bool = True) -> bool:
    return one._mergeable_next(two, mergebool=mergebool)


def mergemany(contents: list[Content]) -> Content:
    assert len(contents) != 0
    return contents[0]._mergemany(contents[1:])


def reduce(
    layout: Content,
    reducer: ak._reducers.Reducer,
    axis: AxisMaybeNone = -1,
    mask: bool = True,
    keepdims: bool = False,
    behavior: dict | None = None,
):
    reducer = layout.backend.prepare_reducer(reducer)

    if axis is None:
        parts = remove_structure(
            layout,
            flatten_records=False,
            drop_nones=False,
            keepdims=keepdims,
            allow_records=True,
            list_to_regular=True,
        )

        if len(parts) > 1:
            # We know that `flatten_records` must fail, so the only other type
            # that can return multiple parts here is the union array
            raise ValueError(
                "cannot use axis=None on an array containing irreducible unions"
            )
        elif len(parts) == 0:
            layout = ak.contents.EmptyArray()
        else:
            (layout,) = parts

        starts = ak.index.Index64.zeros(1, layout.backend.index_nplike)
        parents = ak.index.Index64.zeros(layout.length, layout.backend.index_nplike)
        shifts = None
        next = layout._reduce_next(
            reducer,
            1,
            starts,
            shifts,
            parents,
            1,
            mask,
            keepdims,
            behavior,
        )
        return next[0]
    else:
        negaxis = -axis
        branch, depth = layout.branch_depth

        if branch:
            if negaxis <= 0:
                raise ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves of "
                    "the tree; non-negative from the root)"
                )
            if negaxis > depth:
                raise ValueError(
                    f"cannot use axis={axis} on a nested list structure that splits into "
                    f"different depths, the minimum of which is depth={depth} "
                    "from the leaves"
                )
        else:
            if negaxis <= 0:
                negaxis += depth
            if not 0 < negaxis <= depth:
                raise ValueError(
                    f"axis={axis} exceeds the depth of the nested list structure "
                    f"(which is {depth})"
                )

        starts = ak.index.Index64.zeros(1, layout.backend.index_nplike)
        parents = ak.index.Index64.zeros(layout.length, layout.backend.index_nplike)
        shifts = None
        next = layout._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            1,
            mask,
            keepdims,
            behavior,
        )

        return next[0]


def validity_error(layout: Content, path: str = "layout") -> str:
    return layout._validity_error(path)


def argsort(
    layout: Content,
    axis: int = -1,
    ascending: bool = True,
    stable: bool = False,
) -> Content:
    negaxis = -axis
    branch, depth = layout.branch_depth
    if branch:
        if negaxis <= 0:
            raise ValueError(
                "cannot use non-negative axis on a nested list structure "
                "of variable depth (negative axis counts from the leaves "
                "of the tree; non-negative from the root)"
            )
        if negaxis > depth:
            raise ValueError(
                f"cannot use axis={axis} on a nested list structure that splits into "
                f"different depths, the minimum of which is depth={depth} from the leaves"
            )
    else:
        if negaxis <= 0:
            negaxis = negaxis + depth
        if not 0 < negaxis <= depth:
            raise ValueError(
                f"axis={axis} exceeds the depth of the nested list structure "
                f"(which is {depth})"
            )

    starts = ak.index.Index64.zeros(1, nplike=layout.backend.index_nplike)
    parents = ak.index.Index64.zeros(layout.length, nplike=layout.backend.index_nplike)
    return layout._argsort_next(
        negaxis,
        starts,
        None,
        parents,
        1,
        ascending,
        stable,
    )


def sort(
    layout: Content, axis: int = -1, ascending: bool = True, stable: bool = False
) -> Content:
    negaxis = -axis
    branch, depth = layout.branch_depth
    if branch:
        if negaxis <= 0:
            raise ValueError(
                "cannot use non-negative axis on a nested list structure "
                "of variable depth (negative axis counts from the leaves "
                "of the tree; non-negative from the root)"
            )
        if negaxis > depth:
            raise ValueError(
                f"cannot use axis={axis} on a nested list structure that splits into "
                f"different depths, the minimum of which is depth={depth} from the leaves"
            )
    else:
        if negaxis <= 0:
            negaxis = negaxis + depth
        if not 0 < negaxis <= depth:
            raise ValueError(
                f"axis={axis} exceeds the depth of the nested list structure "
                f"(which is {depth})"
            )

    starts = ak.index.Index64.zeros(1, nplike=layout.backend.index_nplike)
    parents = ak.index.Index64.zeros(layout.length, nplike=layout.backend.index_nplike)
    return layout._sort_next(negaxis, starts, parents, 1, ascending, stable)


def touch_data(layout: Content, recursive: bool = True):
    layout._touch_data(recursive)


def touch_shape(layout: Content, recursive: bool = True):
    layout._touch_shape(recursive)
