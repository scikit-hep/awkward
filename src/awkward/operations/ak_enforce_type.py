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

    if isinstance(type, str):
        type_ = ak.types.from_datashape(type, highlevel=False)

        def select_parameters(type_: ak.types.Type, layout: ak.contents.Content):
            return layout.parameters

    else:

        def select_parameters(type_: ak.types.Type, layout: ak.contents.Content):
            return type_.parameters

        type_ = type

    def recurse(
        type: ak.types.Type, layout: ak.contents.Content
    ) -> ak.contents.Content:
        # Early exit - unknown layouts take the form of the type.
        if layout.is_unknown:
            type_form = ak.forms.from_type(type)
            return type_form.length_zero_array(highlevel=False).copy(
                parameters=select_parameters(type, layout)
            )

        # If we want to lose the option
        elif layout.is_option and not isinstance(type, ak.types.OptionType):
            # Check there are no missing values
            if not layout.backend.index_nplike.all(layout.mask_as_bool(True)):
                raise wrap_error(
                    ValueError("cannot losslessly remove option type from layout")
                )
            return recurse(type, ak.drop_none(layout, axis=0, highlevel=False))

        # Indexed nodes are invisible to layouts
        elif layout.is_indexed:
            return recurse(type, layout.content).copy(
                parameters=select_parameters(type, layout)
            )

        if isinstance(type, ak.types.NumpyType):
            assert layout.is_numpy

            layout = layout.copy(parameters=select_parameters(type, layout))

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
            if layout.is_regular or (
                layout.is_numpy and layout.inner_shape[0] == type.size
            ):
                layout = layout.to_ListOffsetArray64(True)
                return layout.copy(
                    content=recurse(type.content, layout.content),
                    parameters=select_parameters(type, layout),
                )
            elif layout.is_list:
                return layout.copy(
                    content=recurse(type.content, layout.content),
                    parameters=select_parameters(type, layout),
                )
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
            if (layout.is_regular and layout.size == type.size) or (
                layout.is_numpy and layout.inner_shape[0] == type.size
            ):
                return layout.copy(
                    content=recurse(type.content, layout.content),
                    parameters=select_parameters(type, layout),
                )
            elif layout.is_list:
                layout = layout.to_RegularArray()
                return layout.copy(
                    content=recurse(type.content, layout.content),
                    parameters=select_parameters(type, layout),
                )
            else:
                raise wrap_error(
                    AssertionError(f"expected list type, found {type(layout)!r}")
                )

        elif isinstance(type, ak.types.RecordType):
            assert layout.is_record
            assert layout.fields == type.fields  # TODO: do we care about order?
            return layout.copy(
                contents=[
                    recurse(x, y) for x, y in zip(type.contents, layout.contents)
                ],
                parameters=select_parameters(type, layout),
            )

        elif isinstance(type, ak.types.UnionType):
            assert layout.is_union
            return layout.copy(
                contents=[
                    recurse(x, y) for x, y in zip(type.contents, layout.contents)
                ],
                parameters=select_parameters(type, layout),
            )

        elif isinstance(type, ak.types.OptionType):
            if layout.is_option:
                return layout.copy(
                    content=recurse(type.content, layout.content),
                    parameters=select_parameters(type, layout),
                )
            else:
                return ak.contents.UnmaskedArray(recurse(type.content, layout))

        else:
            raise wrap_error(AssertionError(f"unsupported type encountered {type!r}"))

    layout = recurse(type_, layout)
    return wrap_layout(layout, like=array, behavior=behavior, highlevel=highlevel)
