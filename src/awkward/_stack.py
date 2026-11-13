# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Layout traversal shared by tuple/list structure conversions."""

import awkward as ak


def is_string(layout):
    return layout.parameter("__array__") in ("string", "bytestring")


def parameters(layout):
    return {
        key: value
        for key, value in layout.parameters.items()
        if key not in ("__array__", "__record__")
    }


def walk(layout, action, depth=1, path=(), *, rebuild=True):
    # Normalize multidimensional buffers without inspecting their data. Strings
    # are atomic values: their byte lists are not array dimensions.
    if layout.is_numpy and layout.data.ndim > 1:
        layout = layout.to_RegularArray()
    result = action(layout, depth, path)
    if result is not None:
        return result
    if is_string(layout):
        return layout
    if layout.is_list:
        content = walk(layout.content, action, depth + 1, path, rebuild=rebuild)
        return layout.copy(content=content) if rebuild else layout
    if layout.is_record:
        contents = [
            walk(layout.content(field), action, depth, (*path, field), rebuild=rebuild)
            for field in layout.fields
        ]
        return layout.copy(contents=contents) if rebuild else layout
    if layout.is_union:
        contents = [
            walk(content, action, depth, path, rebuild=rebuild)
            for content in layout.contents
        ]
        return (
            simplify_union(
                layout.tags, layout.index, contents, parameters=layout._parameters
            )
            if rebuild
            else layout
        )
    if layout.is_option:
        if not rebuild:
            walk(layout.content, action, depth, path, rebuild=False)
            return layout
        option = layout.to_IndexedOptionArray64()
        return ak.contents.IndexedOptionArray.simplified(
            option.index,
            walk(option.content, action, depth, path),
            parameters=layout._parameters,
        )
    if layout.is_indexed:
        content = walk(layout.content, action, depth, path, rebuild=rebuild)
        return (
            ak.contents.IndexedArray.simplified(
                layout.index, content, parameters=layout._parameters
            )
            if rebuild
            else layout
        )
    return layout


def field_lists(layout):
    """Wrap each field value atomically, using an empty list for missing values."""
    if layout.is_option:
        nplike = layout.backend.nplike
        offsets = nplike.empty(layout.length + 1, dtype="int64")
        offsets[0] = 0
        nplike.cumsum(layout.mask_as_bool(valid_when=True), maybe_out=offsets[1:])
        return ak.contents.ListOffsetArray(ak.index.Index64(offsets), layout.project())
    if layout.is_union:
        return simplify_union(
            layout.tags,
            layout.index,
            [field_lists(content) for content in layout.contents],
        )
    return ak.contents.RegularArray(layout, 1).to_ListOffsetArray64(True)


def simplify_union(tags, index, contents, *, parameters=None):
    # Merging a later branch can make an earlier pair mergeable (for example,
    # lists of integers become compatible with lists of strings once another
    # branch contributes a union of both). Canonicalize until no further
    # branches merge. The number of branches bounds the number of passes.
    previous = len(contents) + 1
    result = ak.contents.UnionArray.simplified(
        tags, index, contents, parameters=parameters
    )
    while result.is_union and len(result.contents) < previous:
        previous = len(result.contents)
        result = ak.contents.UnionArray.simplified(
            result.tags,
            result.index,
            result.contents,
            parameters=result._parameters,
        )
    return result
