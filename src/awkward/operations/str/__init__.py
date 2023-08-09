# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# This set of string-manipulation routines is strongly inspired by Arrow:


# string predicates
# https://arrow.apache.org/docs/python/api/compute.html#string-predicates
from awkward.operations.str.akstr_is_alnum import *
from awkward.operations.str.akstr_is_alpha import *
from awkward.operations.str.akstr_is_decimal import *
from awkward.operations.str.akstr_is_digit import *
from awkward.operations.str.akstr_is_lower import *
from awkward.operations.str.akstr_is_numeric import *
from awkward.operations.str.akstr_is_printable import *
from awkward.operations.str.akstr_is_space import *
from awkward.operations.str.akstr_is_upper import *
from awkward.operations.str.akstr_is_title import *
from awkward.operations.str.akstr_is_ascii import *

# string transforms
# https://arrow.apache.org/docs/python/api/compute.html#string-transforms
from awkward.operations.str.akstr_capitalize import *
from awkward.operations.str.akstr_length import *
from awkward.operations.str.akstr_lower import *
from awkward.operations.str.akstr_swapcase import *
from awkward.operations.str.akstr_title import *
from awkward.operations.str.akstr_upper import *
from awkward.operations.str.akstr_repeat import *
from awkward.operations.str.akstr_replace_slice import *
from awkward.operations.str.akstr_reverse import *
from awkward.operations.str.akstr_replace_substring import *
from awkward.operations.str.akstr_replace_substring_regex import *

# string padding
# https://arrow.apache.org/docs/python/api/compute.html#string-padding
from awkward.operations.str.akstr_center import *
from awkward.operations.str.akstr_lpad import *
from awkward.operations.str.akstr_rpad import *

# string trimming
# https://arrow.apache.org/docs/python/api/compute.html#string-trimming
from awkward.operations.str.akstr_ltrim import *
from awkward.operations.str.akstr_ltrim_whitespace import *
from awkward.operations.str.akstr_rtrim import *
from awkward.operations.str.akstr_rtrim_whitespace import *
from awkward.operations.str.akstr_trim import *
from awkward.operations.str.akstr_trim_whitespace import *

# string splitting
# https://arrow.apache.org/docs/python/api/compute.html#string-splitting
from awkward.operations.str.akstr_split_whitespace import *
from awkward.operations.str.akstr_split_pattern import *
from awkward.operations.str.akstr_split_pattern_regex import *

# string component extraction
# https://arrow.apache.org/docs/python/api/compute.html#string-component-extraction
from awkward.operations.str.akstr_extract_regex import *

# string joining
# https://arrow.apache.org/docs/python/api/compute.html#string-joining
from awkward.operations.str.akstr_join import *
from awkward.operations.str.akstr_join_element_wise import *

# string slicing
# https://arrow.apache.org/docs/python/api/compute.html#string-slicing
from awkward.operations.str.akstr_slice import *

# containment tests
# https://arrow.apache.org/docs/python/api/compute.html#containment-tests
from awkward.operations.str.akstr_count_substring import *
from awkward.operations.str.akstr_count_substring_regex import *
from awkward.operations.str.akstr_ends_with import *
from awkward.operations.str.akstr_find_substring import *
from awkward.operations.str.akstr_find_substring_regex import *
from awkward.operations.str.akstr_index_in import *
from awkward.operations.str.akstr_is_in import *
from awkward.operations.str.akstr_match_like import *
from awkward.operations.str.akstr_match_substring import *
from awkward.operations.str.akstr_match_substring_regex import *
from awkward.operations.str.akstr_starts_with import *

# dictionary-encoding (exclusively for strings)
# https://arrow.apache.org/docs/python/api/compute.html#associative-transforms
from awkward.operations.str.akstr_to_categorical import *


def _get_ufunc_action(
    utf8_function,
    ascii_function,
    *args,
    bytestring_to_string=False,
    **kwargs,
):
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow

    def action(layout, **absorb):
        if layout.is_list and layout.parameter("__array__") == "string":
            return from_arrow(
                utf8_function(to_arrow(layout, extensionarray=False), *args, **kwargs),
                highlevel=False,
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                out = from_arrow(
                    ascii_function(
                        to_arrow(
                            layout.copy(
                                content=layout.content.copy(
                                    parameters={"__array__": "char"}
                                ),
                                parameters={"__array__": "string"},
                            ),
                            extensionarray=False,
                        ),
                        *args,
                        **kwargs,
                    ),
                    highlevel=False,
                )
                if out.is_list and out.parameter("__array__") == "string":
                    out = out.copy(
                        content=out.content.copy(parameters={"__array__": "byte"}),
                        parameters={"__array__": "bytestring"},
                    )
                return out

            else:
                return from_arrow(
                    ascii_function(
                        to_arrow(layout, extensionarray=False), *args, **kwargs
                    ),
                    highlevel=False,
                )

    return action


def _erase_list_option(layout):
    from awkward.contents.unmaskedarray import UnmaskedArray

    assert layout.is_list
    if layout.content.is_option:
        assert isinstance(layout.content, UnmaskedArray)
        return layout.copy(content=layout.content.content)
    else:
        return layout


def _get_split_action(
    utf8_function, ascii_function, *args, bytestring_to_string=False, **kwargs
):
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow

    def action(layout, **absorb):
        if layout.is_list and layout.parameter("__array__") == "string":
            return _erase_list_option(
                from_arrow(
                    utf8_function(
                        to_arrow(layout, extensionarray=False),
                        *args,
                        **kwargs,
                    ),
                    highlevel=False,
                )
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                out = _erase_list_option(
                    from_arrow(
                        ascii_function(
                            to_arrow(
                                layout.copy(
                                    content=layout.content.copy(
                                        parameters={"__array__": "char"}
                                    ),
                                    parameters={"__array__": "string"},
                                ),
                                extensionarray=False,
                            ),
                            *args,
                            **kwargs,
                        ),
                        highlevel=False,
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
                return _erase_list_option(
                    from_arrow(
                        ascii_function(
                            to_arrow(layout, extensionarray=False), *args, **kwargs
                        ),
                        highlevel=False,
                    )
                )

    return action
