# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy
import numpy as np  # noqa: F401

cppyy.add_include_path("src/awkward/_v2/_connect")

compiler = ROOT.gInterpreter.Declare


done = compiler(
    """
#include "rdataframe/ak_array_builders.h"
"""
)
assert done is True

primitive_types = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex128",
    "complex256",
)


def from_rdataframe(data_frame, column, column_as_record=True):
    def _wrap_as_array(column, array, column_as_record):
        return (
            ak._v2.highlevel.Array({column: array})
            if column_as_record
            else ak._v2.highlevel.Array(array)
        )

    def _maybe_wrap(array, column_as_record):
        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[array.layout],
                ),
                highlevel=True,
            )
            if column_as_record
            else array
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)

    ptrs_type = ROOT.awkward.check_type_of[column_type](result_ptrs)

    if ptrs_type in primitive_types:
        form = ak._v2.forms.NumpyForm(ptrs_type)
        ptr, length = ROOT.awkward.copy_buffer[column_type](result_ptrs)
        array = ak._v2.from_buffers(form, length, {"data": ptr}, "data")
        return _maybe_wrap(array, column_as_record)

    elif ptrs_type.startswith("iterable"):

        strings_form = """
{
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "NumpyArray",
        "primitive": "uint8",
        "parameters": {
            "__array__": "char"
        }
    },
    "parameters": {
        "__array__": "string"
    }
}"""
        buffer_form = (
            strings_form
            if ptrs_type[ptrs_type.index(" ") + 1 :] == "chars"
            else str(
                ak._v2.forms.ListOffsetForm(
                    "i64", ak._v2.forms.NumpyForm(ptrs_type[ptrs_type.index(" ") + 1 :])
                )
            )
        )
        result = ROOT.awkward.copy_offsets_and_flatten[column_type](result_ptrs)
        array = ak._v2.from_buffers(
            buffer_form,
            result[0][1],
            {"None-offsets": result[0][0], "None-data": result[1][0]},
        )

        return _maybe_wrap(array, column_as_record)

    elif ptrs_type == "nested iterable":
        # print(ptrs_type)
        #     builder = ak._v2.highlevel.ArrayBuilder()
        #
        #     ROOT.awkward.offsets_and_flatten[column_type](result_ptrs, builder._layout._ptr)
        #
        #     return _maybe_wrap(builder.snapshot(), column_as_record)
        #
        # elif ptrs_type == "awkward type":
        #
        #     # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        #     cpp_reference = result_ptrs.GetValue()
        #
        #     return _wrap_as_array(column, cpp_reference, column_as_record)
        # else:
        raise ak._v2._util.error(NotImplementedError)
