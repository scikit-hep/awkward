# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from awkward.operations.str.ak_is_alnum import *
from awkward.operations.str.ak_is_alpha import *
from awkward.operations.str.ak_is_decimal import *
from awkward.operations.str.ak_is_lower import *


def _get_action(utf8_function, ascii_function, *, bytestring_to_string=False):
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow

    def action(layout, **kwargs):
        if layout.is_list and layout.parameter("__array__") == "string":
            return from_arrow(
                utf8_function(to_arrow(layout, extensionarray=False)), highlevel=False
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                return from_arrow(
                    ascii_function(
                        to_arrow(
                            layout.copy(
                                content=layout.content.copy(
                                    parameters={"__array__": "char"}
                                ),
                                parameters={"__array__": "string"},
                            ),
                            extensionarray=False,
                        )
                    ),
                    highlevel=False,
                )
            else:
                return from_arrow(
                    ascii_function(to_arrow(layout, extensionarray=False)),
                    highlevel=False,
                )

    return action
