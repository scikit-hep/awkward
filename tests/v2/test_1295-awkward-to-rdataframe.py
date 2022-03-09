# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._connect.rdataframe._to_rdataframe  # noqa: E402


ROOT = pytest.importorskip("ROOT")

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402
import awkward._v2._connect.rdataframe._to_rdataframe  # noqa: E402

compiler = ROOT.gInterpreter.Declare


def test_from_awkward_to_rdf():
    v2a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )
    layout = v2a
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)

    generator.generate(compiler)
    generator.generate(print, flatlist_as_rvec=True)
    generator.dataset(flatlist_as_rvec=True)
    print(lookup)

    compiler(
        f"""
void roottest_NumpyArray_v2a(ssize_t length, ssize_t* ptrs) {{
  auto obj = {generator.dataset()};
  std::cout << "Array size " << obj.size() << std::endl;
  for(int64_t i = 0; i < obj.size(); i++)
    std::cout << obj[i] << ", ";
}}
//ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RAwkwardArrayDS> test_MakeAwkwardArrayDataFrame(ULong64_t size) {{
//   auto lm = std::make_unique<ROOT::RDF::RDFDetail::RLoopManager>(std::make_unique<RAwkwardArrayDS>(size), ROOT::RDF::RDFInternal::ColumnNames_t{{"x", "y"}});
//   return ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RAwkwardArrayDS>(std::move(lm));
//}}
//ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RAwkwardArrayDS> test_MakeAwkwardArrayDataFrame() {{
//   auto lm = std::make_unique<ROOT::RDF::RDFDetail::RLoopManager>(std::make_unique<RAwkwardArrayDS>(), ROOT::RDF::RDFInternal::ColumnNames_t{{"x", "y"}});
//   return ROOT::RDF::RInterface<ROOT::RDF::RDFDetail::RLoopManager, RAwkwardArrayDS>(std::move(lm));
//}}
"""
    )
    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    generator = ak._v2._connect.rdataframe._to_rdataframe.togenerator(array.layout.form)
    rdf = generator.generate(
        generator, compiler=compiler, array=array, name="c++func_name"
    )
    # ak._v2._connect.rdataframe._to_rdataframe.generate_RAwkwardArrayDS(
    #     compiler, array, name="c++func_name"
    # )
    ROOT.roottest_NumpyArray_v2a(len(layout), lookup.arrayptrs)


def test_nested_array_1():
    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    # print(array.to_list())
    ak_array_1 = array["x"]
    ak_array_2 = array["y"]
    # print("x:", ak_array_1.to_list(), "y:", ak_array_2.to_list())

    generator = ak._v2._connect.rdataframe._to_rdataframe.togenerator(array.layout.form)
    rdf = generator.generate(
        generator, compiler=compiler, array=array, name="c++func_name"
    )
    # both jitted
    ### rdf = ak._v2.to_rdataframe({"col1": ak_array_1, "col2": ak_array_2})

    # array = ak._v2.from_rdataframe(
    #     rdf, "builder_name", function_for_foreach="my_function"
    # )

    rdf = ROOT.MakeAwkwardDataFrame(
        columns={"col1_name": ak_array_1, "col2_name": ak_array_2}
    )
    rdf.Display().Print()
    # FIXME:
    # rdf_x = rdf.Define("z", lookup.arrayptrs)


@pytest.mark.skip(reason="FIXME: NotImplementedError")
def test_nested_array():
    # array = ak._v2.Array(
    #     [[{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}], [], [{"x": 3, "y": [3.0, 0.3, 3.3]}]]
    #     )
    #
    # array2 = array.x
    #
    # array.show()
    # layout = array.layout
    # >>> lookup.arrayptrs
    # array([             -1, 140408584119136, 140408584119144,               4,
    #                     -1,               3,               8,              10,
    #                     -1, 140408585011824,              -1, 140408584103312,
    #        140408584103320,              14,              -1, 140408583766816])

    # >>> lookup.positions
    # [-1, array([0, 2, 2]), array([2, 2, 3]), 4, -1, 3, 8, 10, -1, array([1, 2, 3]), -1, array([0, 1, 3]), array([1, 3, 6]), 14, -1, array([1.1, 2. , 0.2, 3. , 0.3, 3.3])]

    #     compiler(
    # f'''
    #
    # bool user_cut_function(MyAwkwardSource::entry_type entry) {{
    #     // selects an entry if ALL x > 5
    #     for (size_t i = 0; i < entry.size(); i++) {{
    #         if (entry[i].x() <= 5) {{
    #             return false;
    #         }}
    #     }}
    #     return true;
    # }}
    #
    # bool user_cut_function(MyAwkwardSource::entry_type entry) {{
    #     // selects an entry if ALL x > 5
    #     for (auto subentry : entry) {{
    #         if (subentry.x() <= 5) {{
    #             return false;
    #         }}
    #     }}
    #     return true;
    # }}
    #
    # bool user_cut_function2(MyAwkwardSource2::entry_type entry) {{
    #     return ROOT::RVecOps::All(entry > 5);
    # }}
    #
    #
    # '''
    #     )

    # >>> array.layout.form
    # ListOffsetForm('i64', RecordForm([NumpyForm('int64'), ListOffsetForm('i64', NumpyForm('float64'))], ['x', 'y']))

    v2a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )
    layout = v2a

    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)
    print(lookup.arrayptrs)

    generator.generate(compiler, flatlist_as_rvec=True)

    rvec = ROOT.RVec("ssize_t")(lookup.arrayptrs)
    print(len(layout), rvec)

    # compiler(
    #     f'''
    #     void f({generator.entry_type(flatlist_as_rvec=True)} entry) {{
    #         user writes this
    #     }})
    #
    #     {generator.dataset(length="length", ptrs="arrayptrs", entry="i", flatlist_as_rvec=True)}
    #     '''
    # )

    compiler(
        f"""
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>

void test_nested_rvec_snapshot(ssize_t length, ssize_t* ptrs) {{
   const auto fname = "snapshot_nestedrvecs.root";

   auto obj = {generator.dataset()};
   double* data = reinterpret_cast<double*>(ptrs[1]);

   for( ssize_t i = 0; i < obj.size(); i++) {{
     std::cout << obj[i] << std::endl;

     double value = reinterpret_cast<double*>(ptrs[1])[i];
     std::cout << "Value: " << value <<  " == " << data[i] << std::endl;
   }}

   auto pxs = ROOT::VecOps::RVec<double>(data, obj.size() - 1);
   auto pys = ROOT::VecOps::RVec<double>(data, obj.size() - 1);
   auto Es = ROOT::VecOps::RVec<double>(data, obj.size() - 1);
//{generator.entry_type(flatlist_as_rvec=True)} array;
//array.x
   auto cutPt = [](ROOT::VecOps::RVec<double> &pxs, ROOT::VecOps::RVec<double> &pys, ROOT::VecOps::RVec<double> &Es) {{
    auto all_pts = sqrt(pxs * pxs + pys * pys);
    auto good_pts = all_pts[Es > 3.];
    return good_pts;
   }};

   auto df = ROOT::RDataFrame(1)
                .Define("px",[&] {{
                   return ROOT::VecOps::RVec<double>(data, obj.size());
                }})
                .Define("py",[&] {{
                   return ROOT::VecOps::RVec<double>(data, obj.size());
                }})
                .Define("E",[&] {{
                   return ROOT::VecOps::RVec<double>(data, obj.size());
                }})
                .Define("pt", cutPt, {{"px", "py", "E"}})
                .Define("NumpyArray_array",
                        [&] {{
                           return ROOT::VecOps::RVec<double>(data, obj.size());
                        }});

   auto check = [&](ROOT::RDF::RNode d) {{
      d.Foreach(
         [&](const ROOT::VecOps::RVec<double> &array) {{
            R__ASSERT(All(array == ROOT::VecOps::RVec<double>(data, obj.size())));
            }},
         {{"NumpyArray_array"}});
   }};

   // compiled
   auto out_df1 =
      df.Snapshot<ROOT::VecOps::RVec<double>>("t", fname, {{"NumpyArray_array"}});
   check(*out_df1);

   // jitted
   auto out_df2 = df.Snapshot("t", fname, {{"NumpyArray_array", "px", "py", "E", "pt"}});
   check(*out_df2);
}}
"""
    )

    ROOT.test_nested_rvec_snapshot(len(layout), lookup.arrayptrs)
