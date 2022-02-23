# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def unzip(array, highlevel=True, behavior=None):
    raise NotImplementedError

    # """
    # Args:
    #     array: Array to unzip into individual fields.
    #     highlevel (bool): If True, return an #ak.Array; otherwise, return
    #         a low-level #ak.layout.Content subclass.
    #     behavior (None or dict): Custom #ak.behavior for the output array, if
    #         high-level.

    # If the `array` contains tuples or records, this operation splits them
    # into a Python tuple of arrays, one for each field.

    # If the `array` does not contain tuples or records, the single `array`
    # is placed in a length 1 Python tuple.

    # For example,

    #     >>> array = ak.Array([{"x": 1.1, "y": [1]},
    #     ...                   {"x": 2.2, "y": [2, 2]},
    #     ...                   {"x": 3.3, "y": [3, 3, 3]}])
    #     >>> x, y = ak.unzip(array)
    #     >>> x
    #     <Array [1.1, 2.2, 3.3] type='3 * float64'>
    #     >>> y
    #     <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>
    # """
    # behavior = ak._util.behaviorof(array, behavior=behavior)
    # layout = ak.operations.convert.to_layout(array, allow_record=True, allow_other=False)
    # fields = ak.operations.describe.fields(layout)

    ### FIXME: In v2, you can use return_array=False in Content.recursively_apply
    ###        to perform an action on the whole array but not return a result.
    ###        No need for (brittle) specialized code like the following.

    # def check_for_union(layout):
    #     if isinstance(layout, ak.partition.PartitionedArray):
    #         for x in layout.partitions:
    #             check_for_union(x)

    #     elif isinstance(layout, ak.layout.RecordArray):
    #         pass  # don't descend into nested records

    #     elif isinstance(layout, ak.layout.Record):
    #         pass  # don't descend into nested records

    #     elif isinstance(
    #         layout,
    #         (
    #             ak.layout.UnionArray8_32,
    #             ak.layout.UnionArray8_U32,
    #             ak.layout.UnionArray8_64,
    #         ),
    #     ):
    #         for content in layout.contents:
    #             if set(ak.operations.describe.fields(content)) != set(fields):
    #                 raise ValueError("union of different sets of fields, cannot ak.unzip")

    #     elif hasattr(layout, "content"):
    #         check_for_union(layout.content)

    # check_for_union(layout)

    # if len(fields) == 0:
    #     return (ak._util.maybe_wrap(layout, behavior, highlevel),)
    # else:
    #     return tuple(ak._util.maybe_wrap(layout[n], behavior, highlevel) for n in fields)
