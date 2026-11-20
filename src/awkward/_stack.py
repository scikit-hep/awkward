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
    initial_depth = depth
    multidimensional = layout.is_numpy and layout.data.ndim > 1
    if not multidimensional:
        result = action(layout, depth, path)
        if result is not None:
            return result
        if is_string(layout) or layout.is_leaf:
            return layout

    def apply(node, depth, **kwargs):
        current_depth = initial_depth + depth - 1
        if node is not layout or multidimensional:
            result = action(node, current_depth, path)
            if result is not None:
                return result
        if is_string(node) or node.is_leaf:
            return node
        if node.is_record:
            # Each record field has a separate conversion width. Standard
            # traversal supplies depth, while this adapter carries field paths.
            contents = [
                walk(
                    node.content(field),
                    action,
                    current_depth,
                    (*path, field),
                    rebuild=rebuild,
                )
                for field in node.fields
            ]
            if rebuild and any(
                content is not original
                for content, original in zip(contents, node.contents, strict=True)
            ):
                return node.copy(contents=contents)
            return node
        return None

    # Avoid preparing field slices when record dispatch handles the node.
    if layout.is_record:
        return apply(layout, depth=1)
    result = ak._do.recursively_apply(layout, apply, return_array=rebuild)
    return result if rebuild else layout


def field_lists(layout):
    """Wrap each field value atomically, using an empty list for missing values."""
    if layout.is_union:
        return ak.contents.UnionArray.simplified(
            layout.tags,
            layout.index,
            [field_lists(content) for content in layout.contents],
        )
    return ak.operations.ak_singletons._wrap_singletons(layout, regular=True)
