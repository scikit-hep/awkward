# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("split_pattern",)


import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout

typetracer = TypeTracerBackend.instance()


@high_level_function(module="ak.str")
def split_pattern(
    array, pattern, *, max_splits=None, reverse=False, highlevel=True, behavior=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Pattern of characters/bytes to split on.
        max_splits (None or int): Maximum number of splits for each input
            value. If None, unlimited.
        reverse (bool): If True, start splitting from the end of each input
            value; otherwise, start splitting from the beginning of each
            value. This flag only has an effect if `max_splits` is not None.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Splits any string or bytestring-valued data into a list of substrings
    according to the given separator.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.split_pattern](https://arrow.apache.org/docs/python/generated/pyarrow.compute.split_pattern.html).

    See also: #ak.str.split_whitespace, #ak.str.split_pattern_regex.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, max_splits, reverse, highlevel, behavior)


def _impl(array, pattern, max_splits, reverse, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.split_pattern")
    behavior = behavior_of(array, behavior=behavior)
    layout = ak.to_layout(array)

    if layout.backend is typetracer:
        # FIXME: this workaround for typetracer is required because
        #        split_whitespace does not support length-zero arrays
        #        c.f. https://github.com/apache/arrow/issues/37437
        def action(layout, **kwargs):
            if layout.is_list and layout.parameter("__array__") == "string":
                return (
                    ak.forms.ListOffsetForm(
                        "i32",
                        ak.forms.ListOffsetForm(
                            layout.form.offsets,
                            ak.forms.NumpyForm(
                                "uint8", parameters={"__array__": "char"}
                            ),
                            parameters={"__array__": "string"},
                        ),
                    )
                    .length_zero_array(highlevel=False)
                    .to_typetracer(forget_length=True)
                )

            elif layout.is_list and layout.parameter("__array__") == "bytestring":
                return (
                    ak.forms.ListOffsetForm(
                        "i32",
                        ak.forms.ListOffsetForm(
                            layout.form.offsets,
                            ak.forms.NumpyForm(
                                "uint8", parameters={"__array__": "byte"}
                            ),
                            parameters={"__array__": "bytestring"},
                        ),
                    )
                    .length_zero_array(highlevel=False)
                    .to_typetracer(forget_length=True)
                )

    else:
        action = ak.operations.str._get_split_action(
            pc.split_pattern,
            pc.split_pattern,
            pattern=pattern,
            max_splits=max_splits,
            reverse=reverse,
            bytestring_to_string=False,
        )
    out = ak._do.recursively_apply(layout, action, behavior)

    return wrap_layout(out, behavior, highlevel)
