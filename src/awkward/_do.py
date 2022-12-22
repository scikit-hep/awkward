# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from numbers import Integral

import awkward as ak
from awkward._backends import Backend
from awkward.contents.content import ActionType, Content
from awkward.forms import form
from awkward.record import Record
from awkward.typing import Any, AxisMaybeNone

np = ak._nplikes.NumpyMetadata.instance()


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
) -> Content | Record | None:

    if isinstance(layout, Content):
        return layout._recursively_apply(
            action,
            behavior,
            1,
            copy.copy(depth_context),
            lateral_context,
            {
                "allow_records": allow_records,
                "keep_parameters": keep_parameters,
                "numpy_to_regular": numpy_to_regular,
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
    backend: Backend = None,
) -> tuple[form.Form, int, Mapping[str, Any]]:
    if container is None:
        container = {}
    if backend is None:
        backend = content._backend
    if not backend.nplike.known_data:
        raise ak._errors.wrap_error(
            TypeError("cannot call 'to_buffers' on an array without concrete data")
        )

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
        raise ak._errors.wrap_error(
            TypeError(
                "buffer_key must be a string or a callable, not {}".format(
                    type(buffer_key)
                )
            )
        )

    if form_key is None:
        raise ak._errors.wrap_error(
            TypeError(
                "a 'form_key' must be supplied, to match Form elements to buffers in the 'container'"
            )
        )

    form = content.form_with_key(form_key=form_key, id_start=id_start)

    content._to_buffers(form, getkey, container, backend)

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
        raise ak._errors.wrap_error(
            ValueError("in combinations, 'n' must be at least 1")
        )

    recordlookup = None
    if fields is not None:
        recordlookup = fields
        if len(recordlookup) != n:
            raise ak._errors.wrap_error(
                ValueError("if provided, the length of 'fields' must be 'n'")
            )
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
                    raise ak._errors.wrap_error(
                        np.AxisError(
                            "cannot use non-negative axis on a nested list structure "
                            "of variable depth (negative axis counts from the leaves "
                            "of the tree; non-negative from the root)"
                        )
                    )
                if negaxis > depth:
                    raise ak._errors.wrap_error(
                        np.AxisError(
                            "cannot use axis={} on a nested list structure that splits into "
                            "different depths, the minimum of which is depth={} from the leaves".format(
                                axis, depth
                            )
                        )
                    )
            else:
                if negaxis <= 0:
                    negaxis = negaxis + depth
                if not (0 < negaxis and negaxis <= depth):
                    raise ak._errors.wrap_error(
                        np.AxisError(
                            "axis={} exceeds the depth of this array ({})".format(
                                axis, depth
                            )
                        )
                    )

        starts = ak.index.Index64.zeros(1, nplike=layout._backend.index_nplike)
        parents = ak.index.Index64.zeros(
            layout.length, nplike=layout._backend.index_nplike
        )

        return layout._unique(negaxis, starts, parents, 1)

    raise ak._errors.wrap_error(
        np.AxisError(
            "unique expects axis 'None' or '-1', got axis={} that is not supported yet".format(
                axis
            )
        )
    )


def pad_none(
    layout: Content, length: Integral, axis: Integral, clip: bool = False
) -> Content:
    return layout._pad_none(length, axis, 1, clip)


def completely_flatten(
    layout: Content | Record,
    backend: ak._backends.Backend | None = None,
    flatten_records: bool = True,
    function_name: str | None = None,
    drop_nones: bool = True,
):
    if isinstance(layout, Record):
        return completely_flatten(
            layout._array[layout._at : layout._at + 1],
            backend,
            flatten_records,
            function_name,
            drop_nones,
        )

    else:
        if backend is None:
            backend = layout._backend
        arrays = layout._completely_flatten(
            backend,
            {
                "flatten_records": flatten_records,
                "function_name": function_name,
                "drop_nones": drop_nones,
            },
        )
        return tuple(arrays)


def flatten(layout: Content, axis: Integral = 1) -> Content:
    offsets, flattened = layout._offsets_and_flattened(axis, 1)
    return flattened


def numbers_to_type(layout: Content, name: str) -> Content:
    return layout._numbers_to_type(name)


def fill_none(layout: Content, value: Content) -> Content:
    return layout._fill_none(value)


def num(layout, axis):
    return layout._num(axis, 0)


def mergeable(one: Content, two: Content, mergebool: bool = True) -> bool:
    return one._mergeable(two, mergebool=mergebool)


def merge_as_union(one: Content, two: Content) -> ak.contents.UnionArray:
    mylength = one.length
    theirlength = two.length
    tags = ak.index.Index8.empty((mylength + theirlength), one._backend.index_nplike)
    index = ak.index.Index64.empty((mylength + theirlength), one._backend.index_nplike)
    contents = [one, two]
    assert tags.nplike is one._backend.index_nplike
    one._handle_error(
        one._backend["awkward_UnionArray_filltags_const", tags.dtype.type](
            tags.data, 0, mylength, 0
        )
    )
    assert index.nplike is one._backend.index_nplike
    one._handle_error(
        one._backend["awkward_UnionArray_fillindex_count", index.dtype.type](
            index.data, 0, mylength
        )
    )
    one._handle_error(
        one._backend["awkward_UnionArray_filltags_const", tags.dtype.type](
            tags.data, mylength, theirlength, 1
        )
    )
    one._handle_error(
        one._backend["awkward_UnionArray_fillindex_count", index.dtype.type](
            index.data, mylength, theirlength
        )
    )

    return ak.contents.UnionArray(tags, index, contents, parameters=None)


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
    if axis is None:
        parts = completely_flatten(layout, flatten_records=False, drop_nones=False)

        if len(parts) > 1:
            # We know that `flatten_records` must fail, so the only other type
            # that can return multiple parts here is the union array
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot use axis=None with keepdims=True on an array containing "
                    "irreducible unions"
                )
            )
        elif len(parts) == 0:
            parts = [ak.contents.EmptyArray()]

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
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use non-negative axis on a nested list structure "
                        "of variable depth (negative axis counts from the leaves of "
                        "the tree; non-negative from the root)"
                    )
                )
            if negaxis > depth:
                raise ak._errors.wrap_error(
                    ValueError(
                        "cannot use axis={} on a nested list structure that splits into "
                        "different depths, the minimum of which is depth={} "
                        "from the leaves".format(axis, depth)
                    )
                )
        else:
            if negaxis <= 0:
                negaxis += depth
            if not 0 < negaxis <= depth:
                raise ak._errors.wrap_error(
                    ValueError(
                        "axis={} exceeds the depth of the nested list structure "
                        "(which is {})".format(axis, depth)
                    )
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
    paramcheck = layout._validity_error_parameters(path)
    if paramcheck != "":
        return paramcheck
    return layout._validity_error(path)


def argsort(
    layout: Content,
    axis: int = -1,
    ascending: bool = True,
    stable: bool = False,
    kind: Any = None,
    order: Any = None,
) -> Content:
    negaxis = -axis
    branch, depth = layout.branch_depth
    if branch:
        if negaxis <= 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves "
                    "of the tree; non-negative from the root)"
                )
            )
        if negaxis > depth:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot use axis={} on a nested list structure that splits into "
                    "different depths, the minimum of which is depth={} from the leaves".format(
                        axis, depth
                    )
                )
            )
    else:
        if negaxis <= 0:
            negaxis = negaxis + depth
        if not 0 < negaxis <= depth:
            raise ak._errors.wrap_error(
                ValueError(
                    "axis={} exceeds the depth of the nested list structure "
                    "(which is {})".format(axis, depth)
                )
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
        kind,
        order,
    )


def sort(
    layout: Content,
    axis: int = -1,
    ascending: bool = True,
    stable: bool = False,
    kind: Any = None,
    order: Any = None,
) -> Content:
    negaxis = -axis
    branch, depth = layout.branch_depth
    if branch:
        if negaxis <= 0:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot use non-negative axis on a nested list structure "
                    "of variable depth (negative axis counts from the leaves "
                    "of the tree; non-negative from the root)"
                )
            )
        if negaxis > depth:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot use axis={} on a nested list structure that splits into "
                    "different depths, the minimum of which is depth={} from the leaves".format(
                        axis, depth
                    )
                )
            )
    else:
        if negaxis <= 0:
            negaxis = negaxis + depth
        if not 0 < negaxis <= depth:
            raise ak._errors.wrap_error(
                ValueError(
                    "axis={} exceeds the depth of the nested list structure "
                    "(which is {})".format(axis, depth)
                )
            )

    starts = ak.index.Index64.zeros(1, nplike=layout.backend.index_nplike)
    parents = ak.index.Index64.zeros(layout.length, nplike=layout.backend.index_nplike)
    return layout._sort_next(
        negaxis,
        starts,
        parents,
        1,
        ascending,
        stable,
        kind,
        order,
    )
