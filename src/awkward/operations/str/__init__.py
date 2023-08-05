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


def _get_action(
    utf8_function, ascii_function, *args, bytestring_to_string=False, **kwargs
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
