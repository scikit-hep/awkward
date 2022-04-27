# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def from_rdataframe(data_frame, columns=None, exclude=None, columns_as_records=True):
    def type_of_nested_data(column_type):
        t1 = column_type.rfind("<")
        t2 = column_type.find(">")
        return column_type if t1 == -1 and t2 == -1 else column_type[t1 + 1 : t2]

    # Find all column names in the dataframe
    if not columns:
        columns = [str(x) for x in data_frame.GetColumnNames()]

    # Exclude the specified columns
    if exclude is None:
        exclude = []
    columns = [x for x in columns if x not in exclude]

    if columns_as_records:
        column_type = {}
        type = {}
        result_ptrs = {}
        for col in columns:
            column_type[col] = data_frame.GetColumnType(col)
            type[col] = type_of_nested_data(column_type[col])
            result_ptrs[col] = data_frame.Take[column_type[col]](col)

        # separate it from the above for performance reasons: `GetValue()` triggers
        # an RDF event loop.
        cpp_reference = {}
        for col in columns:
            cpp_reference[col] = result_ptrs[col].GetValue()

    return ak._v2.Array([])
