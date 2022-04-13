# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


@pytest.mark.skip(reason="FIXME: the test fails when flatlist_as_rvec=True")
def test_array_as_rvec():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    layout = array.layout
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)

    generator.generate(compiler, flatlist_as_rvec=True)
    print(lookup.arrayptrs, "DONE!")


# @pytest.mark.skip(reason="FIXME:")
def test_one_array():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    layout = array.layout
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)

    generator.generate(compiler)
    print(lookup.arrayptrs, "DONE!")


def test_simple_test():
    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]
    layout = ak_array_1.layout
    generator = ak._v2._connect.cling.togenerator(layout.form)
    lookup = ak._v2._lookup.Lookup(layout)

    generator.generate(compiler)
    generated_type = generator.entry_type()

    if not hasattr(ROOT, f"get_entry_{generated_type}"):
        done = compiler(
            f"""
        auto get_entry_{generated_type}(ssize_t length, ssize_t* ptrs, int64_t i) {{
            cout << endl << "@@@@@ get entry of {generated_type}..." << endl;
            return {generator.entry()};
        }}
        """.strip()
        )
        assert done is True

    array_view_entry = getattr(ROOT, f"get_entry_{generated_type}")(
        len(layout), lookup.arrayptrs, 0
    )

    if not hasattr(ROOT, f"AwkwardArrayDataSource_{generated_type}"):
        done = compiler(
            f"""
auto erase_array_view = []({type(array_view_entry).__cpp_name__} *p) {{ cout << "Deleter " << endl; }};

class AwkwardArrayDataSource_{generated_type} final : public ROOT::RDF::RDataSource {{
private:
    using PointerHolderPtrs_t = std::vector<ROOT::Internal::TDS::TPointerHolder *>;

    unsigned int fNSlots{{0U}};
    ssize_t column_length;
    ssize_t* column_ptrs;
    {type(array_view_entry).__cpp_name__} fColumn;
    void* fColumnPtr;

    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;

    const PointerHolderPtrs_t fPointerHoldersModels;
    std::vector<PointerHolderPtrs_t> fPointerHolders;

    std::vector<std::pair<ssize_t, ssize_t*>> fColDataPointers;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{{}};

    /// type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {{
        cout << "#2. GetColumnReadersImpl for " << colName;
        auto colNameStr = std::string(colName);
        const auto idName = ROOT::Internal::RDF::TypeID2TypeName(id);
        cout << " and type " << idName << endl;
        auto it = fColTypesMap.find(colNameStr);
        if (fColTypesMap.end() == it) {{
            std::string err = "The specified column name, \"" + colNameStr + "\" is not known to the data source.";
            throw std::runtime_error(err);
        }}

        const auto colIdName = it->second;
        if (colIdName != idName) {{
            std::string err = "Column " + colNameStr + " has type " + colIdName +
                              " while the id specified is associated to type " + idName;
            throw std::runtime_error(err);
        }}

        const auto colBegin = fColNames.begin();
        const auto colEnd = fColNames.end();
        const auto namesIt = std::find(colBegin, colEnd, colName);
        const auto index = std::distance(colBegin, namesIt);

        cout << "index " << index << endl;
        Record_t ret(fNSlots);
        for (auto slot : ROOT::TSeqU(fNSlots)) {{
            cout << "slot " << slot << " and data at " << &fColumn << endl;
            ret[slot] = reinterpret_cast<void*>(&fColumn);
        }}
        return ret;
    }}


    size_t GetEntriesNumber() {{
        cout << "GetEntriesNumber: " << fColNames.size() << endl;
        return fColNames.size();
    }}

public:
    AwkwardArrayDataSource_{generated_type}(
        std::string name,
        std::string column_type,
        ssize_t length,
        ssize_t* ptrs) :
            column_length(length),
            column_ptrs(ptrs),
            fColumn(get_entry_{generated_type}(length, ptrs, 0)),
            fColNames({{name}}),
            fColTypeNames({{column_type}}),
            fColTypesMap( {{ {{ name, column_type }} }})
    {{
        cout << endl << "An AwkwardArrayDataSource_{generated_type} with column names " << endl;
        for (auto n : fColNames) {{
            cout << n << ", ";
        }}
        cout << endl << " and types " << endl;
        for (auto t : fColTypeNames) {{
            cout << t << ", ";
        }}

        for (int64_t i = 0; i < column_length; i++) {{
            auto obj = get_entry_{generated_type}(column_length, column_ptrs, i);
            cout << obj[0] << "(" << &obj << ") == " << fColumn[i] << ", ";
            fColumnPtr = reinterpret_cast<void*>(&obj);
            std::unique_ptr<{type(array_view_entry).__cpp_name__}, decltype(erase_array_view)> obj_ptr(&obj, erase_array_view);
            cout << obj_ptr << endl;
            //fPointerHolders[0][0] = new ROOT::Internal::TDS::TTypedPointerHolder<{type(array_view_entry).__cpp_name__}>(&obj);
        }}
        cout << "is constructed." << endl;
    }}

    ~AwkwardArrayDataSource_{generated_type}() {{
    }}

    void SetNSlots(unsigned int nSlots) {{
        cout << "#1. SetNSlots " << nSlots << " (" << fColumnPtr << ")" << endl;
        fNSlots = nSlots;
        return;
        const auto nCols = fColNames.size();
        fPointerHolders.resize(nCols);

        auto colIndex = 0U;
        for (auto &&ptrHolderv : fPointerHolders) {{
            for (auto slot : ROOT::TSeqI(fNSlots)) {{
                auto ptrHolder = fPointerHoldersModels[colIndex]->GetDeepCopy();
                ptrHolderv.emplace_back(ptrHolder);
                (void)slot;
            }}
            colIndex++;
        }}
        for (auto &&ptrHolder : fPointerHoldersModels)
            delete ptrHolder;
    }}

    void Initialise() {{
        cout << "#3. Initialise" << endl;
        const auto nEntries = GetEntriesNumber();
        cout << "nEntries " << nEntries << endl;
        const auto nEntriesInRange = nEntries / fNSlots; // always one for now
        cout << "nEntriesInRange " << nEntriesInRange << endl;
        auto reminder = 1U == fNSlots ? 0 : nEntries % fNSlots;
        cout << "reminder " << reminder << endl;
        fEntryRanges.resize(fNSlots);
        cout << "fEntryRanges size " << fEntryRanges.size() << endl;
        // FIXME: define some ranges here!
    }}

    const std::vector<std::string> &GetColumnNames() const {{
        return fColNames;
    }}

    bool
    HasColumn(std::string_view colName) const {{
        const auto key = std::string(colName);
        const auto endIt = fColTypesMap.end();
        return endIt != fColTypesMap.find(key);
    }}

    std::string
    GetTypeName(std::string_view colName) const {{
        const auto key = std::string(colName);
        return fColTypesMap.at(key);
    }}

    std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() {{
        cout << "#4. GetEntryRanges" << endl;
        auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
        return entryRanges;
    }}

    bool SetEntry(unsigned int slot, ULong64_t entry) {{
        cout << "#5. SetEntry" << endl;
        return true;
    }}
}};

ROOT::RDataFrame* MakeAwkwardArrayTestDS(std::string name, std::string column_type, ssize_t length, ssize_t* ptrs) {{
    cout << endl << "======= Make AwkwardArray Data Source of {generated_type}!" << endl;
    return new ROOT::RDataFrame(std::make_unique<AwkwardArrayDataSource_{generated_type}>(name, column_type, length, ptrs));
}}
"""
        )
        assert done is True

    data_frame = ROOT.MakeAwkwardArrayTestDS(
        "x", type(array_view_entry).__cpp_name__, len(layout), lookup.arrayptrs
    )
    print("GetColumnType")
    column_type = data_frame.GetColumnType("x")
    print("column_type", column_type)
    print("Take")
    result_ptrs = data_frame.Take[column_type]("x")
    result_ready = result_ptrs.IsReady()
    print("result is ready?", result_ready)


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
