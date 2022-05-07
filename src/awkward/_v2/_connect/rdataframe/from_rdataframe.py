# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

compiler = ROOT.gInterpreter.Declare
numpy = ak.nplike.Numpy.instance()


def from_rdataframe(data_frame, column, column_as_record=True):
    def _wrap_as_array(column, array, column_as_record):
        return (
            ak._v2.highlevel.Array({column: array})
            if column_as_record
            else ak._v2.highlevel.Array(array)
        )

    #
    # def _recurse(cpp_ref):
    #     # Note, the conversion of STL vectors and TVec to numpy arrays in ROOT
    #     # happens without copying the data.
    #     # The memory-adoption is achieved by the dictionary '__array_interface__',
    #     # which is added dynamically to the Python objects by PyROOT.
    #
    #     # '__array_interface__' attribute is added for STL vectors and RVecs of
    #     # the following types:
    #     #   float, double, int, unsigned int, long, unsigned long
    #     if hasattr(cpp_ref, "__array_interface__"):
    #         return cpp_ref
    #     elif (
    #         hasattr(cpp_ref, "begin")
    #         and hasattr(cpp_ref, "end")
    #         and hasattr(cpp_ref, "size")
    #     ):
    #         return cpp_ref #[_recurse(cpp_ref[i]) for i in range(cpp_ref.size())]
    #     # elif isinstance(cpp_ref, cppyy.gbl.variant):
    #     elif isinstance(cpp_ref, complex):
    #         return numpy.asarray(cpp_ref)
    #     else:
    #         raise ak._v2._util.error(NotImplementedError)

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)
    result_ptrs = data_frame_rnode.Take[column_type](column)
    cpp_reference = result_ptrs.GetValue()

    # array = _recurse(cpp_reference)
    return _wrap_as_array(column, cpp_reference, column_as_record)
