# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def to_rdataframe(columns, flatlist_as_rvec=True):
    if not hasattr(ROOT, "awkward_array_columns"):
        done = compiler(
            """
    std::vector<std::pair<std::string, std::string>> awkward_array_columns;
    typedef std::map<std::string, ROOT::Detail::RDF::RColumnReaderBase*> awkward_array_readers_map_type;
    awkward_array_readers_map_type awkward_array_readers_map;
            """
        )
        assert done is True

    if not hasattr(ROOT, "awkward::ArrayWrapper"):
        done = compiler(
            """
    namespace awkward {
        class ArrayWrapper {
        public:
            ArrayWrapper() = delete;
            ArrayWrapper(const ArrayWrapper& wrapper) = delete;
            ArrayWrapper& operator=(ArrayWrapper const& wrapper) = delete;

            ArrayWrapper(
                std::string n,
                std::string t,
                ssize_t l,
                ssize_t* p,
                std::string f,
                std::string et) :
            name(n),
            type(t),
            length(l),
            ptrs(p),
            entry_func(f),
            entry_type(et)
             {
                cout << "ArrayWrapper>>> " <<  this << " is constructed: an ArrayWrapper for an " << name << " RDF column of an array ";
            }

            ~ArrayWrapper() { cout << "........" << this << " ArrayWrapper is destructed." << endl; }

            const std::string name;
            const std::string type;
            const ssize_t length;
            ssize_t* ptrs;
            const std::string entry_func;
            const std::string entry_type;
        };
    }
        """
        )
        assert done is True

    rdf_list_of_columns = []
    rdf_list_of_column_readers = []
    rdf_layouts = {}
    rdf_generators = {}
    rdf_generated_types = {}
    rdf_lookups = {}
    rdf_array_views = {}
    rdf_array_wrappers = {}
    rdf_array_view_entries = {}
    rdf_entry_func = {}
    rdf_column_readers = {}

    for key in columns:
        rdf_layouts[key] = columns[key].layout
        rdf_generators[key] = ak._v2._connect.cling.togenerator(rdf_layouts[key].form)
        rdf_lookups[key] = ak._v2._lookup.Lookup(rdf_layouts[key])

        rdf_generators[key].generate(compiler, flatlist_as_rvec=flatlist_as_rvec)
        generated_type = rdf_generators[key].entry_type()
        rdf_generated_types[key] = generated_type

        if not hasattr(ROOT, f"make_array_{generated_type}_{key}"):
            done = compiler(
                f"""
                auto make_array_{generated_type}_{key}(ssize_t length, ssize_t* ptrs) {{
                    cout << "generated type " << " {generated_type} " << " : key " << " {key} " << endl;
                    return {rdf_generators[key].dataset(flatlist_as_rvec=flatlist_as_rvec)};
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

        if not hasattr(ROOT, f"get_entry_{generated_type}_{key}"):
            done = compiler(
                f"""
            auto get_entry_{generated_type}_{key}(ssize_t length, ssize_t* ptrs, int64_t i) {{
                return {rdf_generators[key].entry(flatlist_as_rvec=flatlist_as_rvec)};
            }}
            """.strip()
            )
            assert done is True
            rdf_entry_func[key] = f"get_entry_{generated_type}_{key}"

            rdf_array_view_entries[key] = getattr(ROOT, rdf_entry_func[key])(
                len(rdf_layouts[key]), rdf_lookups[key].arrayptrs, 0
            )
            print(type(rdf_array_view_entries[key]), rdf_array_view_entries[key][0])

            for i in range(len(rdf_layouts[key])):
                print(rdf_array_view_entries[key][i], ",")

        if not hasattr(
            ROOT, f"awkward::AwkwardArrayColumnReader_{generated_type}_{key}"
        ):
            done = compiler(
                f"""
namespace awkward {{

    class AwkwardArrayColumnReader_{generated_type}_{key} : public ROOT::Detail::RDF::RColumnReaderBase {{
    public:
        AwkwardArrayColumnReader_{generated_type}_{key}(
            {type(rdf_array_view_entries[key]).__cpp_name__} view,
            const std::string entry_name,
            const std::string entry_type,
            ssize_t entry_length,
            ssize_t* entry_ptrs) :
            name(entry_name),
            type(entry_type),
            length(entry_length),
            ptrs(entry_ptrs),
            view_(view) {{
            cout << "CONSTRUCTED AwkwardArrayColumnReader_{generated_type}_{key} of a {type(rdf_array_view_entries[key]).__cpp_name__} at " << &view_ << endl;
            auto obj = {rdf_entry_func[key]}(length, ptrs, 0);
            cout << ROOT::Internal::RDF::TypeID2TypeName(typeid(obj)) << " at " << &obj << endl;
            awkward_array_readers_map["{key}"] = this;
        }}
        ~AwkwardArrayColumnReader_{generated_type}_{key}() {{
            cout << "DESTRUCTED AwkwardArrayColumnReader_{generated_type}_{key} of a {type(rdf_array_view_entries[key]).__cpp_name__} at " << &view_ << endl;
        }}

        const std::string name;
        const std::string type;
        ssize_t length;
        ssize_t* ptrs;

    private:
        void *GetImpl(Long64_t entry) {{
            view_ = {rdf_entry_func[key]}(length, ptrs, entry);
            return reinterpret_cast<void *>(&view_); // FIXME: return view_[entry];
        }}

        {type(rdf_array_view_entries[key]).__cpp_name__} view_;
    }};
}}
    """.strip()
            )
            assert done is True

            rdf_column_readers[key] = getattr(
                ROOT, f"awkward::AwkwardArrayColumnReader_{generated_type}_{key}"
            )(
                rdf_array_view_entries[key],
                f"awkward:{key}",
                type(rdf_array_view_entries[key]).__cpp_name__,
                len(rdf_layouts[key]),
                rdf_lookups[key].arrayptrs,
            )

        rdf_array_wrappers[key] = ROOT.awkward.ArrayWrapper(
            f"awkward:{key}",
            type(
                rdf_array_view_entries[key]
            ).__cpp_name__,  ##type(rdf_array_views[key]).__cpp_name__,
            len(rdf_layouts[key]),
            rdf_lookups[key].arrayptrs,
            rdf_entry_func[key],
            rdf_generated_types[key],
        )

        # rdf_columns[rdf_array_wrappers[key].name] = rdf_array_wrappers[key]
        rdf_list_of_columns.append(rdf_array_wrappers[key])
        rdf_list_of_column_readers.append(rdf_column_readers[key])

    if not hasattr(ROOT, "AwkwardArrayDataSource"):
        cpp_code = f"""
template <typename ...ColumnTypes>
class AwkwardArrayDataSource final : public ROOT::RDF::RDataSource {{
private:

    unsigned int fNSlots{{0U}};
    std::tuple<ColumnTypes*...> fColumns;

    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    const std::vector<std::pair<std::string, std::string>> fColEntryTypes;

    std::vector<std::pair<ssize_t, ssize_t*>> fColDataPointers;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
    std::vector<ROOT::Detail::RDF::RColumnReaderBase*> fColumnReaders;
    const std::map<std::string, ROOT::Detail::RDF::RColumnReaderBase*> fColumnReadersMap;

    /// type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {{
        cout << "#2. GetColumnReadersImpl for " << colName;
        const auto index = std::distance(fColNames.begin(), std::find(fColNames.begin(), fColNames.end(), colName));
        cout << "index " << index << endl;

        return {{}};
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

        //const auto colBegin = fColNames.begin();
        //const auto colEnd = fColNames.end();
        //const auto namesIt = std::find(colBegin, colEnd, colName);
        //const auto index = std::distance(colBegin, namesIt);

        //cout << "index " << index << endl;
        std::size_t t = 0;
        Record_t ret(fNSlots);
        for (auto slot : ROOT::TSeqU(fNSlots)) {{
            cout << "slot " << slot << " and data at ";
            SetColumnPointers(fColumns);
            cout << endl << "will return " << fColDataPointers[index].second << endl;
            cout << "need " << fColDataPointers[index].second;
            cout << endl;
            ret[slot] = fColDataPointers[index].second;
        }}
        return ret;
    }}

    template<typename ... Columns>
    void SetColumnPointers(std::tuple<Columns...> const &columns) {{
    std::size_t length = sizeof...(Columns);
    std::apply(
        [length, this](auto const &...col) {{
            std::cout << "[ ";
            int k = 0;
            ((std::cout << &col), ...);
            std::cout << " ]";
        }},
        columns);
    }}

    size_t GetEntriesNumber() {{
        return std::tuple_size<decltype(fColumns)>::value;
    }}

    // Function to iterate through all values
    // I equals number of values in tuple
    template <size_t I = 0, typename... Ts>
    typename enable_if<I == sizeof...(Ts),
                    void>::type
    printTuple(tuple<Ts...> tup)
    {{
        // If iterated through all values
        // of tuple, then simply return.
        return;
    }}

    template <size_t I = 0, typename... Ts>
    typename enable_if<(I < sizeof...(Ts)),
                    void>::type
    printTuple(tuple<Ts...> tup)
    {{
        // Print element of tuple
        //auto obj = new AwkwardArrayColumnReader(get<I>(tup));
        cout << "done?" << " ";

        // Go to next element
        printTuple<I + 1>(tup);
    }}

public:
    AwkwardArrayDataSource(ColumnTypes&&... wrappers)
        : fColumnReaders(wrappers...),
          fColNames({{wrappers.name...}}),
          fColTypeNames({{wrappers.type...}}),
          fColTypesMap({{ {{wrappers.name, wrappers.type}}...}})
    {{
        cout << endl << "An AwkwardArrayDataSource with column names " << endl;
        cout << "columns number " << GetEntriesNumber() << endl;

        for (int64_t i = 0; i < fColNames.size(); i++) {{
            cout << fColNames[i] << ", ";
        }}
        cout << endl << " and types " << endl;
        for (auto t : fColTypeNames) {{
            cout << t << ", ";
        }}
        cout << "is constructed." << endl;
    }}

    ~AwkwardArrayDataSource() {{
    }}

    void SetNSlots(unsigned int nSlots) {{
        cout << endl
            << "#1. SetNSlots " << nSlots << endl;
        fNSlots = nSlots; // always 1 slot for now
        cout << "SetNSlots done." << endl;
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

template <typename ...ColumnTypes>
ROOT::RDataFrame* MakeAwkwardArrayDS(ColumnTypes&... wrappers) {{
    std::cout << "======= Make AwkwardArray Data Source!" << endl;
    return new ROOT::RDataFrame(std::make_unique<AwkwardArrayDataSource<ColumnTypes...>>(std::move(wrappers)...));
}}
"""
        done = compiler(cpp_code)
        assert done is True

    # rdf = ROOT.MakeAwkwardArrayDS(*rdf_list_of_columns)
    rdf = ROOT.MakeAwkwardArrayDS(*rdf_list_of_column_readers)
    return (
        rdf,
        rdf_array_views,
        rdf_array_wrappers,
        rdf_column_readers,
        rdf_lookups,
        rdf_generators,
    )


def to_rdata_frame(columns, flatlist_as_rvec=True):

    rdf_layouts = {}
    rdf_generators = {}
    rdf_lookups = {}
    rdf_array_views = {}
    rdf_array_wrappers = {}
    rdf_array_view_entries = {}
    rdf_entry_func = {}

    for key in columns:
        rdf_layouts[key] = columns[key].layout
        rdf_generators[key] = ak._v2._connect.cling.togenerator(rdf_layouts[key].form)
        rdf_lookups[key] = ak._v2._lookup.Lookup(rdf_layouts[key])

        rdf_generators[key].generate(compiler, flatlist_as_rvec=flatlist_as_rvec)
        generated_type = rdf_generators[key].entry_type()

        if not hasattr(ROOT, f"make_array_{generated_type}_{key}"):
            done = compiler(
                f"""
                auto make_array_{generated_type}_{key}(ssize_t length, ssize_t* ptrs) {{
                    cout << "generated type " << " {generated_type} " << " : key " << " {key} " << endl;
                    return {rdf_generators[key].dataset(flatlist_as_rvec=flatlist_as_rvec)};
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

        if not hasattr(ROOT, f"get_entry_{generated_type}_{key}"):
            done = compiler(
                f"""
            auto get_entry_{generated_type}_{key}(ssize_t length, ssize_t* ptrs, int64_t i) {{
                return {rdf_generators[key].entry(flatlist_as_rvec=flatlist_as_rvec)};
            }}
            """.strip()
            )
            assert done is True

        rdf_entry_func[key] = f"get_entry_{generated_type}_{key}"
        rdf_array_view_entries[key] = getattr(
            ROOT, f"get_entry_{generated_type}_{key}"
        )(len(rdf_layouts[key]), rdf_lookups[key].arrayptrs, 0)
        print(type(rdf_array_view_entries[key]), rdf_array_view_entries[key][0])

    if not hasattr(ROOT, "AwkwardArrayDataSource"):
        done = compiler(
            f"""
class AwkwardArrayDataSource final : public ROOT::RDF::RDataSource {{
private:

    unsigned int fNSlots{{0U}};
    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    const std::vector<std::pair<std::string, std::string>> fColEntryTypes;
    //std::vector<std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>> fColumnReaders;

    std::vector<std::pair<ssize_t, ssize_t*>> fColDataPointers;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;

    /// type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {{
        cout << "#2. GetColumnReadersImpl for " << colName;
        return {{}};
    }}

    size_t GetEntriesNumber() {{
        return fColNames.size();
    }}

public:
    AwkwardArrayDataSource(std::vector<std::pair<std::string, std::pair<ssize_t, ssize_t*>>> columns)
    {{
        cout << endl << "An AwkwardArrayDataSource with column names " << endl;
    }}

    ~AwkwardArrayDataSource() {{
    }}

    void SetNSlots(unsigned int nSlots) {{
        cout << endl
            << "#1. SetNSlots " << nSlots << endl;
        fNSlots = nSlots; // always 1 slot for now
        cout << "SetNSlots done." << endl;
    }}

    std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
    GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info & /*tid*/) {{
        cout << endl
            << "#2.2. GetColumnReaders " << endl;
        const auto index = std::distance(fColNames.begin(), std::find(fColNames.begin(), fColNames.end(), name));
        return std::move(fColumnReaders[index]);
   }}

    void Initialise() {{
        cout << "#3. Initialise" << endl;
        const auto nEntries = GetEntriesNumber();
        cout << "nEntries " << nEntries << endl;
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

ROOT::RDataFrame* MakeAwkwardArrayDS(std::vector<std::pair<std::string, std::pair<ssize_t, ssize_t*>>> columns) {{
    return new ROOT::RDataFrame(std::make_unique<AwkwardArrayDataSource>(columns));
}}
        template<typename T>
        struct entry {{
            T t;
        }};

        template<typename T>
        entry<typename std::decay<T>::type>
        get_entry(T&& t)
        {{
            return {{ std::forward<T>(t) }};
        }}

"""
        )
        assert done is True

    rdf = ROOT.MakeAwkwardArrayDS()
    return rdf
