# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_categorical(array, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Creates a categorical dataset, which has the following properties:

       * only distinct values (categories) are stored in their entirety,
       * pointers to those distinct values are represented by integers
         (an #ak.contents.IndexedArray or #ak.contents.IndexedOptionArray
         labeled with parameter `"__array__" = "categorical"`.

    This is equivalent to R's "factor", Pandas's "categorical", and
    Arrow/Parquet's "dictionary encoding." It differs from generic uses of
    #ak.contents.IndexedArray and #ak.contents.IndexedOptionArray in Awkward
    Arrays by the guarantee of no duplicate categories and the `"categorical"`
    parameter.

        >>> array = ak.Array([["one", "two", "three"], [], ["three", "two"]])
        >>> categorical = ak.to_categorical(array)
        >>> categorical
        <Array [['one', 'two', 'three'], ..., [...]] type='3 * var * categorical[ty...'>
        >>> categorical.type.show()
        3 * var * categorical[type=string]
        >>> categorical.to_list() == array.to_list()
        True
        >>> ak.categories(categorical)
        <Array ['one', 'two', 'three'] type='3 * string'>
        >>> ak.is_categorical(categorical)
        True
        >>> ak.from_categorical(categorical)
        <Array [['one', 'two', 'three'], ..., ['three', ...]] type='3 * var * string'>

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
            <Array [{x: 1.1, y: 'one'}, ..., {x: 1.1, ...}] type='5 * {x: float64, y: s...'>
        >>> categorical_records = ak.zip({
        ...     "x": ak.to_categorical(records["x"]),
        ...     "y": ak.to_categorical(records["y"]),
        ... })
        >>> categorical_records
        <Array [{x: 1.1, y: 'one'}, ... y: 'one'}] type='5 * {"x": categorical[type=floa...'>
        >>> categorical_records.type.show()
        5 * {
            x: categorical[type=float64],
            y: categorical[type=string]
        }
        >>> categorical_records.to_list() == records.to_list()
        True

    The check for uniqueness is currently implemented in a Python loop, so
    conversion to categorical should be regarded as expensive. (This can
    change, but it would always be an _n log(n)_ operation.)

    See also #ak.is_categorical, #ak.categories, #ak.from_categorical.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_categorical",
        dict(array=array, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, highlevel, behavior)


def _impl(array, highlevel, behavior):
    def action(layout, **kwargs):
        if layout.purelist_depth == 1:
            if layout.is_indexed and layout.is_option:
                content = layout.content
                cls = ak.contents.IndexedOptionArray
            elif layout.is_indexed:
                content = layout.content
                cls = ak.contents.IndexedArray
            elif layout.is_option:
                content = layout.content
                cls = ak.contents.IndexedOptionArray
            else:
                content = layout
                cls = ak.contents.IndexedArray

            content_list = ak.operations.to_list(content)
            hashable = [ak.behaviors.categorical._as_hashable(x) for x in content_list]

            lookup = {}
            is_first = ak._nplikes.numpy.empty(len(hashable), dtype=np.bool_)
            mapping = ak._nplikes.numpy.empty(len(hashable), dtype=np.int64)
            for i, x in enumerate(hashable):
                if x in lookup:
                    is_first[i] = False
                    mapping[i] = lookup[x]
                else:
                    is_first[i] = True
                    lookup[x] = j = len(lookup)
                    mapping[i] = j

            if layout.is_indexed and layout.is_option:
                original_index = ak._nplikes.numpy.asarray(layout.index)
                index = mapping[original_index]
                index[original_index < 0] = -1
                index = ak.index.Index64(index)

            elif layout.is_indexed:
                original_index = ak._nplikes.numpy.asarray(layout.index)
                index = ak.index.Index64(mapping[original_index])

            elif layout.is_option:
                mask = ak._nplikes.numpy.asarray(layout.mask_as_bool(valid_when=False))
                mapping[mask.view(np.bool_)] = -1
                index = ak.index.Index64(mapping)

            else:
                index = ak.index.Index64(mapping)

            out = cls(index, content[is_first], parameters={"__array__": "categorical"})
            return out

        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    out = ak._do.recursively_apply(layout, action, behavior)
    return ak._util.wrap(out, behavior, highlevel)
