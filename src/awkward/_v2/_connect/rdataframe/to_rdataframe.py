# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def to_rdataframe(columns):

    if not hasattr(ROOT, "ArrayWrapper"):
        done = compiler(
            """
        template <typename T>
        class ArrayWrapper {
        public:

            ArrayWrapper() = delete;

            ArrayWrapper(const ArrayWrapper& wrapper) = delete;

            ArrayWrapper& operator=(ArrayWrapper const& wrapper) = delete;

            ArrayWrapper(
                const T& array_view_,
                std::string name_,
                std::string type_,
                ssize_t length_,
                ssize_t* ptrs_) :
            array_view(array_view_),
            name(name_),
            type(type_),
            length(length_),
            ptrs(ptrs_) {
                cout << "ArrayWrapper>>> " <<  this << " is constructed: an ArrayWrapper for an " << name << " RDF column of an array "
                    << ROOT::Internal::RDF::TypeID2TypeName(typeid(&array_view_))
                    << ", size " << array_view.size() << " at " << &array_view << endl;
            }

            ~ArrayWrapper() { cout << "........" << this << " ArrayWrapper of " << &array_view << " is destructed." << endl; }

            const T& array_view;
            const std::string name;
            const std::string type;
            const ssize_t length;
            ssize_t* ptrs;
        };
        """
        )
        assert done is True

    rdf_list_of_columns = []
    rdf_layouts = {}
    rdf_generators = {}
    rdf_lookups = {}
    rdf_array_views = {}
    rdf_array_wrappers = {}

    for key in columns:
        rdf_layouts[key] = columns[key].layout
        rdf_generators[key] = ak._v2._connect.cling.togenerator(rdf_layouts[key].form)
        rdf_lookups[key] = ak._v2._lookup.Lookup(rdf_layouts[key])

        rdf_generators[key].generate(compiler, flatlist_as_rvec=True)
        generated_type = rdf_generators[key].entry_type()

        if not hasattr(ROOT, f"make_array_{generated_type}_{key}"):
            done = compiler(
                f"""
                auto make_array_{generated_type}_{key}(ssize_t length, ssize_t* ptrs) {{
                    return {rdf_generators[key].dataset(flatlist_as_rvec=True)};
                }}
                """.strip()
            )
            assert done is True

        rdf_array_views[key] = getattr(ROOT, f"make_array_{generated_type}_{key}")(
            len(rdf_layouts[key]), rdf_lookups[key].arrayptrs
        )

        print(  # noqa: T001
            "Pass ",
            type(rdf_array_views[key]),
            type(rdf_array_views[key]).__cpp_name__,
            "to an ArrayWrapper...",
        )  # noqa: T001

        rdf_array_wrappers[key] = ROOT.ArrayWrapper[
            type(rdf_array_views[key]).__cpp_name__
        ](
            rdf_array_views[key],
            f"awkward:{key}",
            type(rdf_array_views[key]).__cpp_name__,
            len(rdf_layouts[key]),
            rdf_lookups[key].arrayptrs,
        )

        # rdf_columns[rdf_array_wrappers[key].name] = rdf_array_wrappers[key]
        rdf_list_of_columns.append(rdf_array_wrappers[key])

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
    std::tuple<ArrayWrapper<ColumnTypes>*...> fColumns;
    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    std::vector<std::pair<ssize_t, ssize_t*>> fColDataPointers;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
    std::vector<const void*> fColPtrs;

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
        std::size_t t = 0;
        Record_t ret(fNSlots);
        for (auto slot : ROOT::TSeqU(fNSlots)) {
            cout << "slot " << slot << " and data at ";
            SetColumnPointers(fColumns);
            cout << endl << "will return " << fColDataPointers[index].second << endl;
            cout << "need " << fColDataPointers[index].second;
            cout << endl;
            ret[slot] = fColDataPointers[index].second;
        }
        return ret;
    }

    template<typename ... Columns>
    void SetColumnPointers(std::tuple<Columns...> const &columns) {
    std::size_t length = sizeof...(Columns);
    std::apply(
        [length](auto const &...col) {
            std::cout << "[ ";
            int k = 0;
            ((std::cout << col << (++k == length ? "" : "; ")), ...);
            std::cout << " ]";
        },
        columns);
    }

    size_t GetEntriesNumber() {
        return std::tuple_size<decltype(fColumns)>::value;
    }

public:
    AwkwardArrayDataSource(ArrayWrapper<ColumnTypes>&&... wrappers)
        : fColumns(std::tuple<ArrayWrapper<ColumnTypes>*...>(&wrappers...)),
          fColNames({wrappers.name...}),
          fColTypeNames({wrappers.type...}),
          fColTypesMap({{wrappers.name, wrappers.type}...}),
          fColDataPointers({{wrappers.length, wrappers.ptrs}...}),
          fPointerHoldersModels({new ROOT::Internal::TDS::TTypedPointerHolder<ColumnTypes>(new ColumnTypes())...}) {
        cout << endl << "An AwkwardArrayDataSource with column names " << endl;
        cout << "columns number " << std::tuple_size<decltype(fColumns)>::value << endl;
        for (auto n : fColNames) {
            cout << n << ", ";
        }
        cout << endl << " and types " << endl;
        for (auto t : fColTypeNames) {
            cout << t << ", ";
        }
        cout << "is constructed." << endl;

        cout << "Column data poiners map:" << endl;
        for (auto it : fColDataPointers) {
            cout << it.first << " : " << it.second << endl;
        }
        cout << "GetEntriesNumber " << GetEntriesNumber() << endl;
        SetColumnPointers(fColumns);
    }

    ~AwkwardArrayDataSource() {
    }

    void SetNSlots(unsigned int nSlots) {
        cout << endl
            << "#1. SetNSlots " << nSlots << endl;
        fNSlots = nSlots;
        const auto nCols = fColNames.size();
        fPointerHolders.resize(nCols); // now we need to fill it with the slots, all of the same type
        auto colIndex = 0U;
        for (auto &&ptrHolderv : fPointerHolders) {
            for (auto slot : ROOT::TSeqI(fNSlots)) {
                auto ptrHolder = fPointerHoldersModels[colIndex]->GetDeepCopy();
                ptrHolderv.emplace_back(ptrHolder);
                (void)slot;
            }
        colIndex++;
        }
        for (auto &&ptrHolder : fPointerHoldersModels)
            delete ptrHolder;

        const auto nCols = fColNames.size();
        cout << "size " << nCols << endl;
        auto colIndex = 0U;
        for (auto it : fColDataPointers) {
            cout << "column index " << colIndex++ << endl;
        }
        cout << "SetNSlots done." << endl;
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
ROOT::RDataFrame* MakeAwkwardArrayDS(ArrayWrapper<ColumnTypes>&... wrappers) {
    std::cout << "======= Make AwkwardArray Data Source!" << endl;
    return new ROOT::RDataFrame(std::make_unique<AwkwardArrayDataSource<ColumnTypes...>>(std::move(wrappers)...));
}
"""
        )
        assert done is True

    rdf = ROOT.MakeAwkwardArrayDS(*rdf_list_of_columns)
    return rdf, rdf_array_wrappers, rdf_lookups, rdf_generators
