# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test_nested_layout_builder():
    form = """
{
    "class": "RecordArray",
    "contents": {
        "u": {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "RecordArray",
                "contents": {
                    "i": "int64",
                    "j": {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": "int64"
                    }
                }
            }
        },
        "v": "int64",
        "w": "int64",
        "x": "int64"
    }
}
"""
    builder = ak._v2.highlevel.LayoutBuilder(form)
    #
    #     """
    # {
    #     "form_key": "root",
    #     "class": "RecordArray",
    #     "contents": {
    #         "u": {
    #             "form_key": "u",
    #             "class": "ListOffsetArray",
    #             "offsets": "i64",
    #             "content": {
    #                 "form_key": "u-content",
    #                 "class": "RecordArray",
    #                 "contents": {
    #                     "i": {
    #                         "class": "NumpyArray",
    #                         "primitive": "int64",
    #                         "form_key": "i"
    #                     },
    #                     "j": {
    #                         "class": "ListOffsetArray",
    #                         "offsets": "i64",
    #                         "content": "int64",
    #                         "form_key": "j"
    #                     }
    #                 }
    #             }
    #         },
    #         "v": {
    #             "class": "NumpyArray",
    #             "primitive": "int64",
    #             "form_key": "v"
    #         },
    #         "w": {
    #             "class": "NumpyArray",
    #             "primitive": "int64",
    #             "form_key": "w"
    #         },
    #         "x": {
    #             "class": "NumpyArray",
    #             "primitive": "int64",
    #             "form_key": "x"
    #         }
    #     }
    # }
    # """
    # )

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

    assert ak._v2.to_list(builder.snapshot()) == [
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
    ]


def test_nested_array_builder():
    builder = ak._v2.highlevel.ArrayBuilder()

    builder.begin_record()
    builder.field("u")
    builder.begin_list()  # u

    builder.begin_record()
    builder.field("i")
    builder.integer(1)  # i
    builder.field("j")
    builder.begin_list()  # j
    builder.integer(9)
    builder.integer(8)
    builder.integer(7)
    builder.end_list()  # j
    builder.end_record()

    builder.end_list()  # u

    builder.field("v")
    builder.integer(2)  # v
    builder.field("w")
    builder.integer(3)  # w
    builder.field("x")
    builder.integer(4)  # x
    builder.end_record()

    builder.begin_record()
    builder.field("u")
    builder.begin_list()  # u

    builder.begin_record()
    builder.field("i")
    builder.integer(1)  # i
    builder.field("j")
    builder.begin_list()  # j
    builder.integer(9)
    builder.integer(8)
    builder.integer(7)
    builder.end_list()  # j
    builder.end_record()

    builder.end_list()  # u

    builder.field("v")
    builder.integer(2)  # v
    builder.field("w")
    builder.integer(3)  # w
    builder.field("x")
    builder.integer(4)  # x
    builder.end_record()

    array = builder.snapshot()

    assert ak._v2.to_list(array) == [
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
        {"u": [{"i": 1, "j": [9, 8, 7]}], "v": 2, "w": 3, "x": 4},
    ]
