# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def to_rdataframe(columns):

    if not hasattr(ROOT, "awkward_array_columns"):
        done = compiler(
            """
            std::vector<std::pair<std::string, std::string>> awkward_array_columns;
            std::vector<std::pair<ssize_t, void*>> awkward_array_columns_data;
            typedef std::map<std::string, std::any> awkward_array_map_type;
            awkward_array_map_type awkward_array_columns_map;
            std::map<std::string, void*> awkward_function_map;
            std::map<std::type_index, std::string> awkward_type_name;
            std::map<std::string, std::type_index> awkward_name_type;
            """
        )
        assert done is True
    else:
        ROOT.awkward_array_columns.clear()
        ROOT.awkward_array_columns_data.clear()

        # FIXME: the following are used for debugging only
        ROOT.awkward_array_columns_map.clear()
        ROOT.awkward_function_map.clear()
        ROOT.awkward_type_name.clear()
        ROOT.awkward_name_type.clear()

    if not hasattr(ROOT, "ArrayWrapper"):
        done = compiler(
            """
        template <typename T>
        class ArrayWrapper {
        public:
            ArrayWrapper() = delete;
            ArrayWrapper(std::string name_,
                std::string type_,
                ssize_t length_,
                void* ptrs_) :
            name(name_),
            type(type_),
            length(length_),
            ptrs(ptrs_) {}

            const std::string name;
            const std::string type;
            const ssize_t length;
            void* ptrs;
        };
        """
        )
        assert done is True

    rdf_columns = {}
    rdf_list_of_columns = []

    for key in columns:
        layout = columns[key].layout
        generator = ak._v2._connect.cling.togenerator(layout.form)
        lookup = ak._v2._lookup.Lookup(layout)

        generator.generate(compiler, flatlist_as_rvec=True)
        generated_type = generator.entry_type()

        if not hasattr(ROOT, f"make_array_{generated_type}_{key}"):
            done = compiler(
                f"""
                auto make_array_{generated_type}_{key}(ssize_t length, ssize_t* ptrs) {{
                    auto obj = {generator.dataset(flatlist_as_rvec=True)};
                    awkward_array_columns.push_back({{ "awkward:{key}", ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) }});
                    awkward_array_columns_data.push_back({{length, ptrs}});
                    awkward_array_columns_map[ROOT::Internal::RDF::TypeID2TypeName(typeid(obj))] = &obj;
                    awkward_type_name[std::type_index(typeid(obj))] = ROOT::Internal::RDF::TypeID2TypeName(typeid(obj));
                    awkward_name_type.try_emplace(ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)), std::type_index(typeid(obj)));
                    return obj;
                }}
                """.strip()
            )
            assert done is True

        getattr(ROOT, f"make_array_{generated_type}_{key}")(
            len(layout), lookup.arrayptrs
        )

        arr = ROOT.ArrayWrapper[ROOT.awkward_array_columns[-1][1]](
            ROOT.awkward_array_columns[-1][0],
            ROOT.awkward_array_columns[-1][1],
            ROOT.awkward_array_columns_data[-1][0],
            ROOT.awkward_array_columns_data[-1][1],
        )

        rdf_columns[arr.name] = arr
        rdf_list_of_columns.append(arr)

    if not hasattr(ROOT, "AwkwardArrayDataSource"):
        done = compiler(
            """
template <typename ...ColumnTypes>
class AwkwardArrayDataSource final : public ROOT::RDF::RDataSource {
private:
    using PointerHolderPtrs_t = std::vector<ROOT::Internal::TDS::TPointerHolder *>;
    const PointerHolderPtrs_t fPointerHoldersModels;
    std::vector<PointerHolderPtrs_t> fPointerHolders;

    unsigned int fNSlots{0U};
    std::tuple<ROOT::RVec<ColumnTypes>*...> fColumns;
    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    std::vector<std::pair<ssize_t, void*>> fColDataPointers;
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

    template <std::size_t... S>
    void SetEntryHelper(unsigned int slot, ULong64_t entry, std::index_sequence<S...>) {
        std::initializer_list<int> expander {
            (*static_cast<ColumnTypes *>(fPointerHolders[S][slot]->GetPointer()) = (*std::get<S>(fColumns))[entry], 0)...};
            (void)expander; // avoid unused variable warnings
    }

public:
    AwkwardArrayDataSource(ArrayWrapper<ColumnTypes>... wrappers)
        : fColumns(std::tuple<ROOT::RVec<ColumnTypes>*...>(wrappers.ptrs...)),
          fColNames({wrappers.name...}),
          fColTypeNames({wrappers.type...}),
          fColTypesMap({{wrappers.name, wrappers.type}...}),
          fColDataPointers({{wrappers.length, wrappers.ptrs}...}),
          fPointerHoldersModels({{new ROOT::Internal::TDS::TTypedPointerHolder<ColumnTypes>(new ColumnTypes())...}}) {
        std::cout << std::endl << "An AwkwardArrayDataSource with column names " << std::endl;
        for (auto n : fColNames) {
            std::cout << n << ", ";
        }
        std::cout << std::endl << " and types " << std::endl;
        for (auto t : fColTypeNames) {
            std::cout << t << ", ";
        }
        std::cout << "is constructed." << std::endl;

        std::cout << "Columns map:" << std::endl;
        int n = 0;
        for (auto it : awkward_array_columns) {
            std::cout << it.first << ": " << it.second << " of length " << awkward_array_columns_data[n].first
                << " and data ptrs at " << awkward_array_columns_data[n].second << std::endl;
            auto obj = awkward_array_columns_map[it.first];
            n++;
        }
        std::cout << "Column data poiners map:" << std::endl;
        for (auto it : fColDataPointers) {
            std::cout << it.first << " : " << it.second << std::endl;
        }
        std::cout << "Type name map:" << std::endl;
        for (auto it : awkward_type_name) {
            std::cout << it.first.name() << " : " << it.second << std::endl;
        }
        std::cout << "Name type map:" << std::endl;
        for (auto it : awkward_name_type) {
            std::cout << it.first << " : " << it.second.name() << std::endl;
        }
    }

    ~AwkwardArrayDataSource() {}

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
        SetEntryHelper(slot, entry, std::index_sequence_for<ColumnTypes...>());
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
        assert done is True

    rdf = ROOT.MakeAwkwardArrayDS(*rdf_list_of_columns)
    return rdf
