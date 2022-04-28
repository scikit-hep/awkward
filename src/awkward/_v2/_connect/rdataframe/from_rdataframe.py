# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

compiler = ROOT.gInterpreter.Declare

numpy = ak.nplike.Numpy.instance()


def from_rdataframe(data_frame, column, column_as_record=True):
    # def primitive_form_type(key):
    #     return {
    #         "bool": "bool",
    #         "float": "float32",
    #         "double": "float64",
    #     }[key]

    # FIXME: if columns_as_records:
    column_type = data_frame.GetColumnType(column)

    if "<" in column_type or ">" in column_type:
        raise ak._v2._util.error(NotImplementedError)

    # is it primitive type?
    # print("primitive?", column_type, primitive_form_type(column_type))

    result_ptrs = data_frame.Take[column_type](column)

    # separate it from the above for performance reasons: `GetValue()` triggers
    # an RDF event loop.
    cpp_reference = result_ptrs.GetValue()

    # check that its an std::vector
    if cppyy.typeid(cppyy.gbl.std.vector[column_type]()) == cppyy.typeid(cpp_reference):

        # FIXME: use form?
        # form = ak._v2.forms.numpyform.NumpyForm(primitive_form_type(cpp_reference.value_type))
        array = numpy.asarray(cpp_reference)

        # FIXME: copy data?
        return ak._v2.highlevel.Array(array)

    else:
        raise ak._v2._util.error(NotImplementedError)
