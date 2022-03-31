# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_two_arrays():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]
    ak_array_2 = array["y"]

    # An 'awkward' namespace will be added to the column name
    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}
    )
    # data_frame = ak._v2.operations.convert.to_rdataframe({"x": ak_array_1})
    print("GetColumnType")
    column_type = data_frame.GetColumnType("awkward:x")
    print("column_type", column_type)
    print("Take")
    result_ptrs = data_frame.Take[column_type]("awkward:x")
    result_ready = result_ptrs.IsReady()
    print("result is ready?", result_ready)

    if result_ready:
        ptr = result_ptrs.Get()
        print(">>>pointer", ptr)
        print("result_ptrs", result_ptrs)
        print("GetValue")
        cpp_reference = result_ptrs.GetValue()
        print(cpp_reference)

    done = compiler(
        """
        template <typename Array>
        struct MyFunctor {
            void operator()(const Array& a) {
                cout << a.size() << endl;
            }
        };
        """
    )
    assert done is True
    print("column_type", column_type)
    # f = ROOT.MyFunctor[column_type]()

    # rdf.Foreach(f, ["awkward:x"])


# def test_one_array():
#
#     array = ak._v2.Array(
#         [
#             [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
#             [],
#             [{"x": 3, "y": [3.0, 0.3, 3.3]}],
#         ]
#     )
#     ak_array_1 = array["x"]
#
#     # An 'awkward' namespace will be added to the column name
#     rdf = ak._v2.operations.convert.to_rdataframe({"xxx": ak_array_1})
#     print(rdf)
#
#
# def test_rdf():
#     array = ak._v2.Array(
#         [
#             [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
#             [],
#             [{"x": 3, "y": [3.0, 0.3, 3.3]}],
#         ]
#     )
#     ak_array_1 = array["x"]
#
#     layout = ak_array_1.layout
#     generator = ak._v2._connect.cling.togenerator(layout.form)
#     lookup = ak._v2._lookup.Lookup(layout)
#
#     generator.generate(compiler, flatlist_as_rvec=True)
#     generated_type = generator.entry_type()
#     key = "x"
#
#     if not hasattr(ROOT, f"make_array_column_{generated_type}_{key}"):
#         done = compiler(
#             f"""
#             auto make_array_column_{generated_type}_{key}(ssize_t length, ssize_t* ptrs) {{
#                 auto obj = {generator.dataset(flatlist_as_rvec=True)};
#                 awkward_array_columns.push_back({{ "awkward:{key}", ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) }});
#                 awkward_array_columns_data.push_back({{length, ptrs}});
#                 awkward_array_columns_map[ROOT::Internal::RDF::TypeID2TypeName(typeid(obj))] = &obj;
#                 awkward_type_name[std::type_index(typeid(obj))] = ROOT::Internal::RDF::TypeID2TypeName(typeid(obj));
#                 awkward_name_type.try_emplace(ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)), std::type_index(typeid(obj)));
#                 return obj;
#             }}
#             """.strip()
#         )
#         assert done is True
#
#     f = getattr(ROOT, f"make_array_column_{generated_type}_{key}")(
#         len(layout), lookup.arrayptrs
#     )
#     # length = {len(layout)}
#     done = compiler(
#         f"""
#         auto df = ROOT::RDataFrame(1)
#             .Define("awkward.{key}", [] {{
#                     return {f};
#                 }});
#         """
#         )
#     assert done is True
