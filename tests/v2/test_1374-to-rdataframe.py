# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_array_wrapper():
    ROOT.gInterpreter.ProcessLine(
        """
    std::vector<std::pair<std::string, std::string>> awkward_array_columns;
    typedef std::map<std::string, std::any> awkward_array_map_type;
    awkward_array_map_type awkward_array_columns_map;
    """
    )

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]
    ak_array_2 = array["y"]

    columns = {"col1_name_x": ak_array_1, "col2_name_y": ak_array_2}
    rdf_columns = {}

    for key in columns:
        layout = columns[key].layout
        generator = ak._v2._connect.cling.togenerator(layout.form)
        lookup = ak._v2._lookup.Lookup(layout)

        generator.generate(compiler, flatlist_as_rvec=True)
        generated_type = generator.entry_type()
        cpp_type = "class " + generated_type + "; "

        print(key, cpp_type)
        print(generator.dataset(), generator.entry())
        err = compiler(
            f"""
            auto make_array_{generated_type}(ssize_t length, ssize_t* ptrs) {{
                auto obj = {generator.dataset(flatlist_as_rvec=True)};
                std::cout << "constructed " << " {generated_type} " << " == "
                    << ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) << std::endl;
                awkward_array_columns.push_back({{ " {key} ", ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) }});
                awkward_array_columns_map[ROOT::Internal::RDF::TypeID2TypeName(typeid(obj))] = obj;
                return obj;
            }}
            """
        )
        assert err is True

        f = getattr(ROOT, f"make_array_{generated_type}")(len(layout), lookup.arrayptrs)

        rdf_columns[key] = type(f)
        print(rdf_columns)

    ROOT.gInterpreter.Declare(
        """
template <typename T>
class ArrayWrapper {
    std::string name;
};

template <typename ...ColumnTypes>
class AwkwardArrayDataSource final : public ROOT::RDF::RDataSource {
private:
    using PointerHolderPtrs_t = std::vector<ROOT::Internal::TDS::TPointerHolder *>;
    const PointerHolderPtrs_t fPointerHoldersModels;
    std::vector<PointerHolderPtrs_t> fPointerHolders;

    unsigned int fNSlots{0U};
    const std::vector<std::string> fColNames;
    std::tuple<ColumnTypes...> fColumnTypes;
    const std::map<std::string, std::string> fColTypesMap;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;

    /// type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {
        auto colNameStr = std::string(colName);
        const auto idName = ROOT::Internal::RDF::TypeID2TypeName(id);
        auto it = fColTypesMap.find(colNameStr);
        if (fColTypesMap.end() == it) {
            std::string err = "The specified column name, \"" + colNameStr + "\" is not known to the data source.";
            throw std::runtime_error(err);
        }

        const auto colIdName = it->second;
        if (colIdName != idName) {
            std::string err = "Column " + colNameStr + " has type " + colIdName +
                              " while the id specified is associated to type " + idName;
            throw std::runtime_error(err);
        }

        const auto colBegin = fColNames.begin();
        const auto colEnd = fColNames.end();
        const auto namesIt = std::find(colBegin, colEnd, colName);
        const auto index = std::distance(colBegin, namesIt);

        Record_t ret(fNSlots);
        for (auto slot : ROOT::TSeqU(fNSlots)) {
            ret[slot] = fPointerHolders[index][slot]->GetPointerAddr();
        }
        return ret;
    }

public:
    AwkwardArrayDataSource(ArrayWrapper<ColumnTypes>... wrappers)
        : fColNames({wrappers.name...}),
          fColumnTypes({wrappers...}){
        for (auto name : fColNames) {
            std::cout << name << ", ";
        }
        std::cout << std::endl;
        using type_tuple = std::tuple<ColumnTypes...>;
        std::cout << "tuple of types " << std::tuple_size_v<type_tuple> << std::endl;
        std::cout << std::endl;

        std::cout << "Constructed " << std::endl;
        for (auto it : awkward_array_columns) {
            std::cout << it.first << ": " << it.second << std::endl;
        }
    }

    void SetNSlots(unsigned int nSlots) {
        fNSlots = nSlots;
    }

    const std::vector<std::string> &GetColumnNames() const {
        return fColNames;
    }

    bool
    HasColumn(std::string_view colName) const {
        const auto key = std::string(colName);
        const auto endIt = fColTypesMap.end();
        return endIt != fColTypesMap.find(key);
    }

    std::string
    GetTypeName(std::string_view colName) const {
        const auto key = std::string(colName);
        return fColTypesMap.at(key);
    }

    std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() {
        auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
        return entryRanges;
    }

    bool SetEntry(unsigned int slot, ULong64_t entry) {
        // FIXME: SetEntryHelper(slot, entry, std::index_sequence_for<ColumnTypes...>());
        return true;
    }
};

template <typename ...ColumnTypes>
ROOT::RDataFrame* MakeAwkwardArrayDS(ArrayWrapper<ColumnTypes>... wrappers) {
    return new ROOT::RDataFrame(std::make_unique<AwkwardArrayDataSource<ColumnTypes...>>(
        std::forward<ArrayWrapper<ColumnTypes>>(wrappers)...
    ));
}
"""
    )
    arr1 = ROOT.ArrayWrapper[ROOT.awkward_array_columns[0][1]]()
    arr1.name = ROOT.awkward_array_columns[0][0]
    print(">>>", arr1.name, arr1)

    arr2 = ROOT.ArrayWrapper[ROOT.awkward_array_columns[1][1]]()
    arr2.name = ROOT.awkward_array_columns[1][0]
    print(">>>", arr2.name, arr2)

    f = ROOT.MakeAwkwardArrayDS(arr1, arr2)
    print("HERE:", type(f))
