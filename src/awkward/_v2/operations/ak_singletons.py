# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def singletons(array, highlevel=True, behavior=None):
    """
    Args:
        array: Data to wrap in lists of length 1 if present and length 0
            if missing (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns a singleton list (length 1) wrapping each non-missing value and
    an empty list (length 0) in place of each missing value.

    For example,

        >>> array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
        >>> print(ak.singletons(array))
        [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]

    See #ak.firsts to invert this function.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.singletons",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, **kwargs):
        nplike = ak.nplike.of(layout)

        if layout.is_OptionType:
            nulls = nplike.index_nplike.asarray(
                layout.mask_as_bool(valid_when=False)
            ).view(np.bool_)
            offsets = nplike.index_nplike.ones(len(layout) + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:][nulls] = 0
            nplike.index_nplike.cumsum(offsets, out=offsets)
            return ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(offsets), layout.project()
            )

        elif isinstance(layout, ak._v2.contents.IndexedArray) and isinstance(
            layout.content, (ak._v2.contents.EmptyArray, ak._v2.contents.NumpyArray)
        ):
            return action(
                ak._v2.contents.IndexedOptionArray(
                    layout.index,
                    layout.content,
                    layout.identifier,
                    layout.parameters,
                )
            )

        elif isinstance(layout, ak._v2.contents.EmptyArray):
            return action(ak._v2.contents.UnmaskedArray(layout.toNumpyArray(np.int64)))

        elif isinstance(layout, ak._v2.contents.NumpyArray):
            return action(ak._v2.contents.UnmaskedArray(layout))

        else:
            return None

    layout = ak._v2.operations.to_layout(array)
    out = layout.recursively_apply(action, behavior)

    return ak._v2._util.wrap(out, behavior, highlevel)
