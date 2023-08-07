# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# https://arrow.apache.org/docs/python/api/compute.html#string-predicates

# string predicates
from awkward.operations.str.ak_is_alnum import *
from awkward.operations.str.ak_is_alpha import *
from awkward.operations.str.ak_is_decimal import *
from awkward.operations.str.ak_is_digit import *
from awkward.operations.str.ak_is_lower import *
from awkward.operations.str.ak_is_numeric import *
from awkward.operations.str.ak_is_printable import *
from awkward.operations.str.ak_is_space import *
from awkward.operations.str.ak_is_upper import *
from awkward.operations.str.ak_is_title import *
from awkward.operations.str.ak_is_ascii import *

# string transforms
from awkward.operations.str.ak_capitalize import *
from awkward.operations.str.ak_length import *
from awkward.operations.str.ak_lower import *
from awkward.operations.str.ak_swapcase import *
from awkward.operations.str.ak_title import *
from awkward.operations.str.ak_upper import *
from awkward.operations.str.ak_repeat import *
from awkward.operations.str.ak_replace_slice import *
from awkward.operations.str.ak_reverse import *
from awkward.operations.str.ak_replace_substring import *
from awkward.operations.str.ak_replace_substring_regex import *

# string padding
from awkward.operations.str.ak_center import *
from awkward.operations.str.ak_lpad import *
from awkward.operations.str.ak_rpad import *

# string trimming
from awkward.operations.str.ak_ltrim import *
from awkward.operations.str.ak_ltrim_whitespace import *
from awkward.operations.str.ak_rtrim import *
from awkward.operations.str.ak_rtrim_whitespace import *
from awkward.operations.str.ak_trim import *
from awkward.operations.str.ak_trim_whitespace import *

# string splitting
from awkward.operations.str.ak_split_whitespace import *
from awkward.operations.str.ak_split_pattern import *
from awkward.operations.str.ak_split_pattern_regex import *

# string component extraction

from awkward.operations.str.ak_extract_regex import *

# string joining

from awkward.operations.str.ak_join import *
from awkward.operations.str.ak_join_element_wise import *

# string slicing

from awkward.operations.str.ak_slice import *

# containment tests

from awkward.operations.str.ak_count_substring import *
from awkward.operations.str.ak_count_substring_regex import *
from awkward.operations.str.ak_ends_with import *


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
