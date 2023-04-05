# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("enforce_type",)
import awkward as ak
from awkward._errors import wrap_error
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.types.numpytype import primitive_to_dtype

np = NumpyMetadata.instance()


def enforce_type(
    array,
    type,
    *,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        type (#ak.types.Type or str): The type of the Awkward
            Array to enforce.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.


    """
    with ak._errors.OperationErrorContext(
        "ak.enforce_type",
        {
            "array": array,
            "type": type,
            "highlevel": highlevel,
            "behavior": behavior,
        },
    ):
        return _impl(array, type, highlevel, behavior)


def _impl(array, type, highlevel, behavior):
    layout = ak.to_layout(array)
    type_ = (
        ak.types.from_datashape(type, highlevel=False)
        if isinstance(type, str)
        else type
    )
    layout = _recurse(type_, layout)
    return wrap_layout(layout, like=array, behavior=behavior, highlevel=highlevel)


def _recurse(type: ak.types.Type, layout: ak.contents.Content) -> ak.contents.Content:
    if layout.is_indexed:
        return _recurse(type, layout.content)

    # Take parameters from type (excepting strings, categoricals?, don't break those!)

    if layout.is_unknown:
        type_form: ak.forms.Form = ak.forms.from_type(type)
        return type_form.length_zero_array()  # TODO: length?

    if isinstance(type, ak.types.NumpyType):
        assert layout.is_numpy

        dtype = primitive_to_dtype(type.primitive)
        if np.issubdtype(layout.dtype, dtype):
            return layout
        else:
            return ak.values_astype(layout, dtype, highlevel=False)

    elif isinstance(type, ak.types.ListType):
        assert (
            layout.is_list
            or layout.is_regular
            or (layout.is_numpy and len(layout.shape) > 1)
        )
        if layout.is_regular:
            layout = layout.to_ListOffsetArray64(True)
            return layout.copy(content=_recurse(type.content, layout.content))
        elif layout.is_numpy and layout.inner_shape[0] == type.size:
            layout = layout.to_ListOffsetArray64(True)
            return layout.copy(content=_recurse(type.content, layout.content))
        elif layout.is_list:
            return layout.copy(content=_recurse(type.content, layout.content))
        else:
            raise wrap_error(
                AssertionError(f"expected list type, found {type(layout)!r}")
            )

    elif isinstance(type, ak.types.RegularType):
        assert (
            layout.is_list
            or layout.is_regular
            or (layout.is_numpy and len(layout.shape) > 1)
        )
        if layout.is_regular and layout.size == type.size:
            return layout.copy(content=_recurse(type.content, layout.content))
        elif layout.is_numpy and layout.inner_shape[0] == type.size:
            return layout.copy(content=_recurse(type.content, layout.content))
        elif layout.is_list:
            layout = layout.to_RegularArray()
            return layout.copy(content=_recurse(type.content, layout.content))
        else:
            raise wrap_error(
                AssertionError(f"expected list type, found {type(layout)!r}")
            )

    elif isinstance(type, ak.types.RecordType):
        assert layout.is_record
        assert layout.fields == type.fields  # TODO: do we care about order?
        return layout.copy(
            contents=[_recurse(x, y) for x, y in zip(type.contents, layout.contents)]
        )

    elif isinstance(type, ak.types.UnionType):
        assert layout.is_union
        return layout.copy(
            contents=[_recurse(x, y) for x, y in zip(type.contents, layout.contents)]
        )

    elif isinstance(type, ak.types.OptionType):
        if layout.is_option:
            return layout.copy(content=_recurse(type.content, layout.content))
        else:
            return ak.contents.UnmaskedArray(_recurse(type.content, layout))

    else:
        raise wrap_error(AssertionError(f"unsupported type encountered {type!r}"))
