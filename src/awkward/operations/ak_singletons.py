# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def singletons(array, highlevel=True, behavior=None):
    """
    Args:
        array: Data to wrap in lists of length 1 if present and length 0
            if missing (None).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
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
    with ak._errors.OperationErrorContext(
        "ak.singletons",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, **kwargs):
        nplike = ak.nplikes.nplike_of(layout)

        if layout.is_option:
            nulls = nplike.index_nplike.asarray(
                layout.mask_as_bool(valid_when=False)
            ).view(np.bool_)
            offsets = nplike.index_nplike.ones(len(layout) + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:][nulls] = 0
            nplike.index_nplike.cumsum(offsets, out=offsets)
            return ak.contents.ListOffsetArray(
                ak.index.Index64(offsets), layout.project()
            )

        elif isinstance(layout, ak.contents.IndexedArray) and isinstance(
            layout.content, (ak.contents.EmptyArray, ak.contents.NumpyArray)
        ):
            return action(
                ak.contents.IndexedOptionArray(
                    layout.index,
                    layout.content,
                    layout.parameters,
                )
            )

        elif isinstance(layout, ak.contents.EmptyArray):
            return action(ak.contents.UnmaskedArray(layout.toNumpyArray(np.int64)))

        elif isinstance(layout, ak.contents.NumpyArray):
            return action(ak.contents.UnmaskedArray(layout))

        else:
            return None

    layout = ak.operations.to_layout(array)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    out = layout.recursively_apply(action, behavior)

    return ak._util.wrap(out, behavior, highlevel)
