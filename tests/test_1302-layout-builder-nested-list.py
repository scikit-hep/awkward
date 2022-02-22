# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    builder = ak.layout.LayoutBuilder64(
        """
    {
        "form_key": "root",
        "class": "RecordArray",
        "contents": {
            "u": {
                "form_key": "u",
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "form_key": "u-content",
                    "class": "RecordArray",
                    "contents": {
                        "i": {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "form_key": "i"
                        },
                        "j": {
                            "class": "ListOffsetArray64",
                            "offsets": "i64",
                            "content": "int64",
                            "form_key": "j"
                        }
                    }
                }
            },
            "v": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "v"
            },
            "w": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "w"
            },
            "x": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "x"
            }
        }
    }
    """
    )

    builder.begin_list()  # u

    builder.int64(1)  # i
    builder.begin_list()  # j
    builder.int64(9)
    builder.int64(8)
    builder.int64(7)
    builder.end_list()  # j

    builder.end_list()  # u

    builder.int64(2)  # v
    builder.int64(3)  # w
    builder.int64(4)  # x

    builder.begin_list()  # u

    builder.int64(1)  # i
    builder.begin_list()  # j
    builder.int64(9)
    builder.int64(8)
    builder.int64(7)
    builder.end_list()  # j

    builder.end_list()  # u

    builder.int64(2)  # v
    builder.int64(3)  # w
    builder.int64(4)  # x

    assert ak.to_list(builder.snapshot()) == [
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
    ]
