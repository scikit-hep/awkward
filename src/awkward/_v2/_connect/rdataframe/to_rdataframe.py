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
            ArrayWrapper(
                const T& ak_view_,
                std::string name_,
                std::string type_,
                ssize_t length_,
                void* ptrs_) :
            ak_view(&ak_view_),
            view_ptr(reinterpret_cast<const void*>(&ak_view_)),
            name(name_),
            type(type_),
            length(length_),
            ptrs(ptrs_) {
                cout << "Constructed an ArrayWrapper for an " << name << " RDF column of an array " << ROOT::Internal::RDF::TypeID2TypeName(typeid(ak_view_))
                    << ", size " << ak_view->size() << " at " << ak_view
                    << "(" << view_ptr << ")"<< endl;
            }

            const T* ak_view;
            const void* view_ptr;
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
                    cout << endl << "Make an " << ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) << " at " <<  &obj << endl;
                    return obj;
                }}
                """.strip()
            )
            assert done is True

        ak_view = getattr(ROOT, f"make_array_{generated_type}_{key}")(
            len(layout), lookup.arrayptrs
        )
        print("Pass ", type(ak_view), "to an ArrayWrapper...")  # noqa: T001

        arr = ROOT.ArrayWrapper[ROOT.awkward_array_columns[-1][1]](
            ak_view,
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
        cout << "#2. GetColumnReadersImpl for " << colName;
        auto colNameStr = std::string(colName);
        const auto idName = ROOT::Internal::RDF::TypeID2TypeName(id);
        cout << " and type " << idName << endl;
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
        cout << "index " << index << endl;
        Record_t ret(fNSlots);
        for (auto slot : ROOT::TSeqU(fNSlots)) {
            cout << "slot " << slot << endl;
            ret[slot] = fColDataPointers[index].second;
        }
        return ret;
    }

    template<typename ... Ts>
    void output_tuple(std::tuple<Ts...> const &tpl) {
    std::size_t length = sizeof...(Ts);
    std::apply(
        [length](auto const &...ps) {
            std::cout << "[ ";
            int k = 0;
            ((std::cout << ps << (++k == length ? "" : "; ")), ...);
            std::cout << " ]";
        },
        tpl);
    }

   size_t GetEntriesNumber() { return fColNames.size(); }

public:
    AwkwardArrayDataSource(ArrayWrapper<ColumnTypes>... wrappers)
        : fColumns(std::tuple<ROOT::RVec<ColumnTypes>*...>(wrappers...)),
          fColNames({wrappers.name...}),
          fColTypeNames({wrappers.type...}),
          fColTypesMap({{wrappers.name, wrappers.type}...}),
          fColDataPointers({{wrappers.length, wrappers.ptrs}...}) {
        cout << endl << "An AwkwardArrayDataSource with column names " << endl;
        for (auto n : fColNames) {
            cout << n << ", ";
        }
        cout << endl << " and types " << endl;
        for (auto t : fColTypeNames) {
            cout << t << ", ";
        }
        cout << "is constructed." << endl;

        cout << "Columns map:" << endl;
        int n = 0;
        for (auto it : awkward_array_columns) {
            cout << it.first << ": " << it.second << " of length " << awkward_array_columns_data[n].first
                << " and data ptrs at " << awkward_array_columns_data[n].second << endl;
            auto obj = awkward_array_columns_map[it.first];
            n++;
        }
        cout << "Column data poiners map:" << endl;
        for (auto it : fColDataPointers) {
            cout << it.first << " : " << it.second << endl;
        }
        cout << "Type name map:" << endl;
        for (auto it : awkward_type_name) {
            cout << it.first.name() << " : " << it.second << endl;
        }
        cout << "Name type map:" << endl;
        for (auto it : awkward_name_type) {
            cout << it.first << " : " << it.second.name() << endl;
        }
        output_tuple(fColumns);
    }

    ~AwkwardArrayDataSource() {}

    void SetNSlots(unsigned int nSlots) {
        cout << "#1. SetNSlots " << nSlots << endl;
        fNSlots = nSlots;
        const auto nCols = fColNames.size();
        cout << "size " << nCols << endl;
        auto colIndex = 0U;
        for (auto it : fColDataPointers) {
            cout << "column index " << colIndex++ << endl;
        }
    }

    void Initialise() {
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
        cout << "#4. GetEntryRanges" << endl;
        auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
        return entryRanges;
    }

    bool SetEntry(unsigned int slot, ULong64_t entry) {
        cout << "#5. SetEntry" << endl;
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
