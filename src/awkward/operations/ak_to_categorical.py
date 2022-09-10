# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_categorical(array, highlevel=True):
    """
    Args:
        array: Data convertible to an Awkward Array
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    Creates a categorical dataset, which has the following properties:

       * only distinct values (categories) are stored in their entirety,
       * pointers to those distinct values are represented by integers
         (an #ak.layout.IndexedArray or #ak.layout.IndexedOptionArray
         labeled with parameter `"__array__" = "categorical"`.

    This is equivalent to R's "factor", Pandas's "categorical", and
    Arrow/Parquet's "dictionary encoding." It differs from generic uses of
    #ak.layout.IndexedArray and #ak.layout.IndexedOptionArray in Awkward
    Arrays by the guarantee of no duplicate categories and the `"categorical"`
    parameter.

        >>> array = ak.Array([["one", "two", "three"], [], ["three", "two"]])
        >>> categorical = ak.to_categorical(array)
        >>> categorical
        <Array [['one', 'two', ... 'three', 'two']] type='3 * var * categorical[type=str...'>
        >>> ak.type(categorical)
        3 * var * categorical[type=string]
        >>> ak.to_list(categorical) == ak.to_list(array)
        True
        >>> ak.categories(categorical)
        <Array ['one', 'two', 'three'] type='3 * string'>
        >>> ak.is_categorical(categorical)
        True
        >>> ak.from_categorical(categorical)
        <Array [['one', 'two', ... 'three', 'two']] type='3 * var * string'>

    This function descends through nested lists, but not into the fields of
    records, so records can be categories. To make categorical record
    fields, split up the record, apply this function to each desired field,
    and #ak.zip the results together.

        >>> records = ak.Array([
        ...     {"x": 1.1, "y": "one"},
        ...     {"x": 2.2, "y": "two"},
        ...     {"x": 3.3, "y": "three"},
        ...     {"x": 2.2, "y": "two"},
        ...     {"x": 1.1, "y": "one"}
        ... ])
        >>> records
        <Array [{x: 1.1, y: 'one'}, ... y: 'one'}] type='5 * {"x": float64, "y": string}'>
        >>> categorical_records = ak.zip({
        ...     "x": ak.to_categorical(records["x"]),
        ...     "y": ak.to_categorical(records["y"]),
        ... })
        >>> categorical_records
        <Array [{x: 1.1, y: 'one'}, ... y: 'one'}] type='5 * {"x": categorical[type=floa...'>
        >>> ak.type(categorical_records)
        5 * {"x": categorical[type=float64], "y": categorical[type=string]}
        >>> ak.to_list(categorical_records) == ak.to_list(records)
        True

    The check for uniqueness is currently implemented in a Python loop, so
    conversion to categorical should be regarded as expensive. (This can
    change, but it would always be an _n log(n)_ operation.)

    See also #ak.is_categorical, #ak.categories, #ak.from_categorical.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_categorical",
        dict(array=array, highlevel=highlevel),
    ):
        return _impl(array, highlevel)


def _impl(array, highlevel):
    def action(layout, **kwargs):
        if layout.purelist_depth == 1:
            if layout.is_OptionType:
                layout = layout.simplify_optiontype()
            if layout.is_IndexedType and layout.is_OptionType:
                content = layout.content
                cls = ak._v2.contents.IndexedOptionArray
            elif layout.is_IndexedType:
                content = layout.content
                cls = ak._v2.contents.IndexedArray
            elif layout.is_OptionType:
                content = layout.content
                cls = ak._v2.contents.IndexedOptionArray
            else:
                content = layout
                cls = ak._v2.contents.IndexedArray

            content_list = ak._v2.operations.to_list(content)
            hashable = [
                ak._v2.behaviors.categorical.as_hashable(x) for x in content_list
            ]

            lookup = {}
            is_first = ak.nplike.numpy.empty(len(hashable), dtype=np.bool_)
            mapping = ak.nplike.numpy.empty(len(hashable), dtype=np.int64)
            for i, x in enumerate(hashable):
                if x in lookup:
                    is_first[i] = False
                    mapping[i] = lookup[x]
                else:
                    is_first[i] = True
                    lookup[x] = j = len(lookup)
                    mapping[i] = j

            if layout.is_IndexedType and layout.is_OptionType:
                original_index = ak.nplike.numpy.asarray(layout.index)
                index = mapping[original_index]
                index[original_index < 0] = -1
                index = ak._v2.index.Index64(index)

            elif layout.is_IndexedType:
                original_index = ak.nplike.numpy.asarray(layout.index)
                index = ak._v2.index.Index64(mapping[original_index])

            elif layout.is_OptionType:
                mask = ak.nplike.numpy.asarray(layout.mask_as_bool(valid_when=False))
                mapping[mask.view(np.bool_)] = -1
                index = ak._v2.index.Index64(mapping)

            else:
                index = ak._v2.index.Index64(mapping)

            out = cls(index, content[is_first], parameters={"__array__": "categorical"})
            return out

        else:
            return None

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    out = layout.recursively_apply(action)
    if highlevel:
        out = ak._v2._util.wrap(out, ak._v2._util.behavior_of(array))
        return out
    else:
        return out
