# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
from awkward._v2.highlevel import Array

np = ak.nplike.NumpyMetadata.instance()


class CategoricalBehavior(Array):
    __name__ = "Array"


class _HashableDict:
    def __init__(self, obj):
        self.keys = tuple(sorted(obj))
        self.values = tuple(_hashable(obj[k]) for k in self.keys)
        self.hash = hash((_HashableDict,) + self.keys, self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableDict)
            and self.keys == other.keys
            and self.values == other.values
        )


class _HashableList:
    def __init__(self, obj):
        self.values = tuple(obj)
        self.hash = hash((_HashableList,) + self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return isinstance(other, _HashableList) and self.values == other.values


def _hashable(obj):
    if isinstance(obj, dict):
        return _HashableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(_hashable(x) for x in obj)
    elif isinstance(obj, list):
        return _HashableList(obj)
    else:
        return obj


def _categorical_equal(one, two):
    behavior = ak._v2._util.behavior_of(one, two)

    one, two = one.layout, two.layout

    assert one.is_IndexedType or (one.is_OptionType and one.is_IndexedType)
    assert two.is_IndexedType or (two.is_OptionType and two.is_IndexedType)
    assert one.parameter("__array__") == "categorical"
    assert two.parameter("__array__") == "categorical"

    one_index = ak.nplike.numpy.asarray(one.index)
    two_index = ak.nplike.numpy.asarray(two.index)
    one_content = ak._v2._util.wrap(one.content, behavior)
    two_content = ak._v2._util.wrap(two.content, behavior)

    if len(one_content) == len(two_content) and ak._v2.operations.all(
        one_content == two_content, axis=None
    ):
        one_mapped = one_index

    else:
        one_list = ak._v2.operations.to_list(one_content)
        two_list = ak._v2.operations.to_list(two_content)
        one_hashable = [_hashable(x) for x in one_list]
        two_hashable = [_hashable(x) for x in two_list]
        two_lookup = {x: i for i, x in enumerate(two_hashable)}

        one_to_two = ak.nplike.numpy.empty(len(one_hashable) + 1, dtype=np.int64)
        for i, x in enumerate(one_hashable):
            one_to_two[i] = two_lookup.get(x, len(two_hashable))
        one_to_two[-1] = -1

        one_mapped = one_to_two[one_index]

    out = one_mapped == two_index
    out = ak._v2._util.wrap(
        ak._v2.contents.NumpyArray(out), ak._v2._util.behavior_of(one, two)
    )
    return out


def _apply_ufunc(ufunc, method, inputs, kwargs):
    nextinputs = []
    for x in inputs:
        if isinstance(x, ak._v2.highlevel.Array) and x.layout.is_IndexedType:
            nextinputs.append(
                ak._v2.highlevel.Array(
                    x.layout.project(), behavior=ak._v2._util.behavior_of(x)
                )
            )
        else:
            nextinputs.append(x)

    return getattr(ufunc, method)(*nextinputs, **kwargs)


def is_categorical(array):
    """
    Args:
        array: A possibly-categorical Awkward Array.

    If the `array` is categorical (contains #ak.layout.IndexedArray or
    #ak.layout.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns True;
    otherwise, it returns False.

    See also #ak.categories, #ak.to_categorical, #ak.from_categorical.
    """

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    return layout.purelist_parameter("__array__") == "categorical"


def categories(array, highlevel=True):
    """
    Args:
        array: A possibly-categorical Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    If the `array` is categorical (contains #ak.layout.IndexedArray or
    #ak.layout.IndexedOptionArray labeled with parameter
    `"__array__" = "categorical"`), then this function returns its categories.

    See also #ak.is_categorical, #ak.to_categorical, #ak.from_categorical.
    """

    output = [None]

    def action(layout, **kwargs):
        if layout.parameter("__array__") == "categorical":
            output[0] = layout.content
            return layout

        else:
            return None

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    layout.recursively_apply(action)

    if output[0] is None:
        return None
    elif highlevel:
        return ak._v2._util.wrap(output[0], ak._v2._util.behavior_of(array))
    else:
        return output[0]


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
            hashable = [_hashable(x) for x in content_list]

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


def from_categorical(array, highlevel=True):
    """
    Args:
        array: Awkward Array from which to remove the 'categorical' parameter.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.

    This function replaces categorical data with non-categorical data (by
    removing the label that declares it as such).

    This is a metadata-only operation; the running time does not scale with the
    size of the dataset. (Conversion to categorical is expensive; conversion
    from categorical is cheap.)

    See also #ak.is_categorical, #ak.categories, #ak.to_categorical,
    #ak.from_categorical.
    """

    def action(layout, **kwargs):
        if layout.parameter("__array__") == "categorical":
            out = ak._v2.operations.with_parameter(
                layout, "__array__", None, highlevel=False
            )
            return out

        else:
            return None

    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    out = layout.recursively_apply(action)
    if highlevel:
        return ak._v2._util.wrap(out, ak._v2._util.behavior_of(array))
    else:
        return out


def register(behavior):
    behavior["categorical"] = CategoricalBehavior
    behavior[ak.nplike.numpy.equal, "categorical", "categorical"] = _categorical_equal
    behavior[ak.nplike.numpy.ufunc, "categorical"] = _apply_ufunc
