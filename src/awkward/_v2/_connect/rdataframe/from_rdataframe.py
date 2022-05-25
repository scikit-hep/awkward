# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

cppyy.add_include_path("include")
cppyy.add_include_path("src/awkward/_v2/_connect")

compiler = ROOT.gInterpreter.Declare


done = compiler(
    """
#include "rdataframe/ak_array_builders.h"
"""
)
assert done is True


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
    builder = ak._v2.highlevel.ArrayBuilder()

    ptrs_type = ROOT.awkward.check_type_of[column_type](
        result_ptrs, builder._layout._ptr
    )

    if ptrs_type in ("primitive", "complex", "iterable", "iterable of complex"):

        return _maybe_wrap(builder.snapshot(), column_as_record)

    elif ptrs_type == "nested iterable":

        ROOT.awkward.offsets_and_flatten[column_type](result_ptrs, builder._layout._ptr)

        return _maybe_wrap(builder.snapshot(), column_as_record)

    elif ptrs_type == "awkward type":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, cpp_reference, column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
