# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# This set of string-manipulation routines is strongly inspired by Arrow:

# string transforms
# https://arrow.apache.org/docs/python/api/compute.html#string-transforms
from __future__ import annotations
from awkward.operations.str.akstr_capitalize import *

# string padding
# https://arrow.apache.org/docs/python/api/compute.html#string-padding
from awkward.operations.str.akstr_center import *

# containment tests
# https://arrow.apache.org/docs/python/api/compute.html#containment-tests
from awkward.operations.str.akstr_count_substring import *
from awkward.operations.str.akstr_count_substring_regex import *
from awkward.operations.str.akstr_ends_with import *

# string component extraction
# https://arrow.apache.org/docs/python/api/compute.html#string-component-extraction
from awkward.operations.str.akstr_extract_regex import *
from awkward.operations.str.akstr_find_substring import *
from awkward.operations.str.akstr_find_substring_regex import *
from awkward.operations.str.akstr_index_in import *

# string predicates
# https://arrow.apache.org/docs/python/api/compute.html#string-predicates
from awkward.operations.str.akstr_is_alnum import *
from awkward.operations.str.akstr_is_alpha import *
from awkward.operations.str.akstr_is_ascii import *
from awkward.operations.str.akstr_is_decimal import *
from awkward.operations.str.akstr_is_digit import *
from awkward.operations.str.akstr_is_in import *
from awkward.operations.str.akstr_is_lower import *
from awkward.operations.str.akstr_is_numeric import *
from awkward.operations.str.akstr_is_printable import *
from awkward.operations.str.akstr_is_space import *
from awkward.operations.str.akstr_is_title import *
from awkward.operations.str.akstr_is_upper import *

# string joining
# https://arrow.apache.org/docs/python/api/compute.html#string-joining
from awkward.operations.str.akstr_join import *
from awkward.operations.str.akstr_join_element_wise import *
from awkward.operations.str.akstr_length import *
from awkward.operations.str.akstr_lower import *
from awkward.operations.str.akstr_lpad import *

# string trimming
# https://arrow.apache.org/docs/python/api/compute.html#string-trimming
from awkward.operations.str.akstr_ltrim import *
from awkward.operations.str.akstr_ltrim_whitespace import *
from awkward.operations.str.akstr_match_like import *
from awkward.operations.str.akstr_match_substring import *
from awkward.operations.str.akstr_match_substring_regex import *
from awkward.operations.str.akstr_repeat import *
from awkward.operations.str.akstr_replace_slice import *
from awkward.operations.str.akstr_replace_substring import *
from awkward.operations.str.akstr_replace_substring_regex import *
from awkward.operations.str.akstr_reverse import *
from awkward.operations.str.akstr_rpad import *
from awkward.operations.str.akstr_rtrim import *
from awkward.operations.str.akstr_rtrim_whitespace import *

# string slicing
# https://arrow.apache.org/docs/python/api/compute.html#string-slicing
from awkward.operations.str.akstr_slice import *
from awkward.operations.str.akstr_split_pattern import *
from awkward.operations.str.akstr_split_pattern_regex import *

# string splitting
# https://arrow.apache.org/docs/python/api/compute.html#string-splitting
from awkward.operations.str.akstr_split_whitespace import *
from awkward.operations.str.akstr_starts_with import *
from awkward.operations.str.akstr_swapcase import *
from awkward.operations.str.akstr_title import *

# dictionary-encoding (exclusively for strings)
# https://arrow.apache.org/docs/python/api/compute.html#associative-transforms
from awkward.operations.str.akstr_to_categorical import *
from awkward.operations.str.akstr_trim import *
from awkward.operations.str.akstr_trim_whitespace import *
from awkward.operations.str.akstr_upper import *


def _drop_option_preserving_form(layout, ensure_empty_mask: bool = False):
    from awkward._do import recursively_apply
    from awkward.contents import UnmaskedArray, IndexedOptionArray, IndexedArray

    def action(_, continuation, **kwargs):
        this = continuation()

        if not this.is_option:
            return this
        elif isinstance(this, UnmaskedArray):
            return this.content
        else:
            index_nplike = this.backend.index_nplike
            assert not (
                ensure_empty_mask
                and index_nplike.known_data
                and index_nplike.any(this.mask_as_bool(valid_when=False))
            ), "did not expect option type, but arrow returned a non-erasable option"
            # Re-write indexed options as indexed
            if isinstance(this, IndexedOptionArray):
                return IndexedArray(
                    this.index, this.content, parameters=this._parameters
                )
            else:
                return this.content

    return recursively_apply(layout, action, return_simplified=False)


def _apply_through_arrow(
    operation,
    /,
    *args,
    generate_bitmasks=False,
    expect_option_type=False,
    string_to32=False,
    bytestring_to32=False,
    ensure_empty_mask=False,
    **kwargs,
):
    from awkward._backends.dispatch import backend_of
    from awkward.contents.content import Content
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow
    from awkward._backends.typetracer import TypeTracerBackend

    backend = backend_of(*args)

    typetracer = TypeTracerBackend.instance()
    if backend is typetracer:
        converted_args = [
            to_arrow(
                x.form.length_zero_array(),
                extensionarray=False,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
            )
            if isinstance(x, Content)
            else x
            for x in args
        ]
        out = from_arrow(
            operation(*converted_args, **kwargs),
            generate_bitmasks=generate_bitmasks,
            highlevel=False,
        ).to_typetracer(forget_length=True)

    else:
        converted_args = [
            to_arrow(
                x,
                extensionarray=False,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
            )
            if isinstance(x, Content)
            else x
            for x in args
        ]
        out = from_arrow(
            operation(*converted_args, **kwargs),
            generate_bitmasks=generate_bitmasks,
            highlevel=False,
        )

    if expect_option_type:
        return out
    else:
        return _drop_option_preserving_form(out, ensure_empty_mask=ensure_empty_mask)


def _get_ufunc_action(
    utf8_function,
    ascii_function,
    *args,
    generate_bitmasks=False,
    bytestring_to_string=False,
    expect_option_type=False,
    **kwargs,
):
    def action(layout, **absorb):
        if layout.is_list and layout.parameter("__array__") == "string":
            return _apply_through_arrow(
                utf8_function,
                layout,
                *args,
                generate_bitmasks=generate_bitmasks,
                expect_option_type=expect_option_type,
                **kwargs,
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                out = _apply_through_arrow(
                    ascii_function,
                    layout.copy(
                        content=layout.content.copy(parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    *args,
                    generate_bitmasks=generate_bitmasks,
                    expect_option_type=expect_option_type,
                    **kwargs,
                )
                if out.is_list and out.parameter("__array__") == "string":
                    out = out.copy(
                        content=out.content.copy(parameters={"__array__": "byte"}),
                        parameters={"__array__": "bytestring"},
                    )
                return out

            else:
                return _apply_through_arrow(
                    ascii_function,
                    layout,
                    *args,
                    generate_bitmasks=generate_bitmasks,
                    expect_option_type=expect_option_type,
                    **kwargs,
                )

    return action


def _get_split_action(
    utf8_function,
    ascii_function,
    *args,
    generate_bitmasks=False,
    bytestring_to_string=False,
    **kwargs,
):
    from awkward._backends.typetracer import TypeTracerBackend
    from awkward.forms import ListOffsetForm, NumpyForm

    typetracer = TypeTracerBackend.instance()

    # FIXME: this workaround for typetracer is required because
    #        split_XXX does not support length-zero arrays
    #        c.f. https://github.com/apache/arrow/issues/37437
    def action(layout, **_):
        if layout.backend is typetracer:
            if layout.is_list and layout.parameter("__array__") == "string":
                return (
                    ListOffsetForm(
                        "i32",
                        ListOffsetForm(
                            layout.form.offsets,
                            NumpyForm("uint8", parameters={"__array__": "char"}),
                            parameters={"__array__": "string"},
                        ),
                    )
                    .length_zero_array()
                    .to_typetracer(forget_length=True)
                )

            elif layout.is_list and layout.parameter("__array__") == "bytestring":
                return (
                    ListOffsetForm(
                        "i32",
                        ListOffsetForm(
                            layout.form.offsets,
                            NumpyForm("uint8", parameters={"__array__": "byte"}),
                            parameters={"__array__": "bytestring"},
                        ),
                    )
                    .length_zero_array()
                    .to_typetracer(forget_length=True)
                )
        else:
            if layout.is_list and layout.parameter("__array__") == "string":
                return _drop_option_preserving_form(
                    _apply_through_arrow(
                        utf8_function,
                        layout,
                        *args,
                        generate_bitmasks=generate_bitmasks,
                        **kwargs,
                    )
                )

            elif layout.is_list and layout.parameter("__array__") == "bytestring":
                if bytestring_to_string:
                    out = _drop_option_preserving_form(
                        _apply_through_arrow(
                            ascii_function,
                            layout.copy(
                                content=layout.content.copy(
                                    parameters={"__array__": "char"}
                                ),
                                parameters={"__array__": "string"},
                            ),
                            *args,
                            generate_bitmasks=generate_bitmasks,
                            **kwargs,
                        )
                    )
                    assert out.is_list

                    assert (
                        out.content.is_list
                        and out.content.parameter("__array__") == "string"
                    )
                    return out.copy(
                        content=out.content.copy(
                            content=out.content.content.copy(
                                parameters={"__array__": "byte"}
                            ),
                            parameters={"__array__": "bytestring"},
                        ),
                    )

                else:
                    return _drop_option_preserving_form(
                        _apply_through_arrow(
                            ascii_function,
                            layout,
                            *args,
                            generate_bitmasks=generate_bitmasks,
                            **kwargs,
                        )
                    )

    return action
