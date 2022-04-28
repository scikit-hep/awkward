# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

compiler = ROOT.gInterpreter.Declare

numpy = ak.nplike.Numpy.instance()

cppyy.cppdef(
    """
#include <type_traits>

template<typename T>
bool _is_arithmetic() {
    return std::is_arithmetic<T>::value;
};
"""
)


def from_rdataframe(data_frame, column, column_as_record=True):

    column_type = data_frame.GetColumnType(column)
    result_ptrs = data_frame.Take[column_type](column)
    cpp_reference = result_ptrs.GetValue()

    # check that its an std::vector
    if cppyy.typeid(cppyy.gbl.std.vector[column_type]()) == cppyy.typeid(cpp_reference):

        # check if it's an integral or a floating point type
        if cppyy.gbl._is_arithmetic[cpp_reference.value_type]():
            # FIXME: use form?
            # form = ak._v2.forms.numpyform.NumpyForm(primitive_form_type(cpp_reference.value_type))
            array = numpy.asarray(cpp_reference)

            # FIXME: copy data?
            return (
                ak._v2.highlevel.Array({column: array})
                if column_as_record
                else ak._v2.highlevel.Array(array)
            )
        else:
            raise ak._v2._util.error(NotImplementedError)

    else:
        raise ak._v2._util.error(NotImplementedError)
