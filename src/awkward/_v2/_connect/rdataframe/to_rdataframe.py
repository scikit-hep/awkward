# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def to_rdataframe(columns, flatlist_as_rvec=True):

    if not hasattr(ROOT, "awkward::ArrayWrapper"):
        done = compiler(
            """
    namespace awkward {
        class ArrayWrapper {
        public:
            ArrayWrapper() = delete;
            ArrayWrapper(const ArrayWrapper& wrapper) = delete;
            ArrayWrapper& operator=(ArrayWrapper const& wrapper) = delete;

            // An array wrapper is short-lived.
            // It gets destructed after AwkwardArrayDataSource::SetNSlots are done.
            ArrayWrapper(
                const std::string_view name,
                const std::string_view type,
                ssize_t length,
                ssize_t* ptrs) :
            name(name),
            type(type),
            length(length),
            ptrs(ptrs) {}

            const std::string_view name;
            const std::string_view type;
            const ssize_t length;
            ssize_t* ptrs;
        };
    }
        """
        )
        assert done is True

    # FIXME: check the need for all these dictionaries
    rdf_layouts = {}
    rdf_generators = {}
    rdf_generated_types = {}
    rdf_lookups = {}
    rdf_list_of_wrappers = []
    rdf_array_wrappers = {}
    rdf_array_view_entries = {}
    rdf_entry_func = {}
    rdf_column_readers = {}
    rdf_array_data_source_class_name = "AwkwardArrayDataSource_of"

    for key in columns:
        rdf_layouts[key] = columns[key].layout
        rdf_generators[key] = ak._v2._connect.cling.togenerator(rdf_layouts[key].form)
        rdf_lookups[key] = ak._v2._lookup.Lookup(rdf_layouts[key])

        rdf_generators[key].generate(compiler, flatlist_as_rvec=flatlist_as_rvec)
        generated_type = rdf_generators[key].entry_type()
        rdf_generated_types[key] = generated_type

        # Generate a unique class name
        rdf_array_data_source_class_name = (
            rdf_array_data_source_class_name + f"""_{generated_type}_as_{key}"""
        )

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

        rdf_array_wrappers[key] = ROOT.awkward.ArrayWrapper(
            f"awkward:{key}",
            type(rdf_array_view_entries[key]).__cpp_name__,
            len(rdf_layouts[key]),
            rdf_lookups[key].arrayptrs,
        )

        if not hasattr(
            ROOT, f"awkward::AwkwardArrayColumnReader_{generated_type}_{key}"
        ):
            done = compiler(
                f"""
namespace awkward {{
    auto erase_array_view_{generated_type}_{key} = []({type(rdf_array_view_entries[key]).__cpp_name__} *entry) {{ cout << "Adoid deleter of " << entry << endl; }};
    // FIXME: need a deleter?
    // std::unique_ptr<{type(rdf_array_view_entries[key]).__cpp_name__}, decltype(erase_array_view_{generated_type})> obj_ptr(&obj, erase_array_view_{generated_type});

    class AwkwardArrayColumnReader_{generated_type}_{key} : public ROOT::Detail::RDF::RColumnReaderBase {{
    public:
        AwkwardArrayColumnReader_{generated_type}_{key}(ssize_t length, ssize_t* ptrs)
            : length_(length),
              ptrs_(ptrs),
              view_(get_entry_{generated_type}_{key}(length, ptrs, 0)) {{
            cout << "CONSTRUCTED AwkwardArrayColumnReader_{generated_type}_{key} of a {type(rdf_array_view_entries[key]).__cpp_name__} at " << &view_ << endl;
        }}
        ~AwkwardArrayColumnReader_{generated_type}_{key}() {{
            cout << "DESTRUCTED AwkwardArrayColumnReader_{generated_type}_{key} of a {type(rdf_array_view_entries[key]).__cpp_name__} at " << &view_ << endl;
        }}

        ssize_t length_;
        ssize_t* ptrs_;

    private:
        void* GetImpl(Long64_t entry) {{
            view_ = get_entry_{generated_type}_{key}(length_, ptrs_, entry);
            return reinterpret_cast<void*>(&view_);
        }}

        {type(rdf_array_view_entries[key]).__cpp_name__} view_;
    }};
}}
    """.strip()
            )
            assert done is True

        rdf_list_of_wrappers.append(rdf_array_wrappers[key])

    if not hasattr(ROOT, rdf_array_data_source_class_name):
        cpp_code_begin = f"""
template <typename ...ColumnTypes>
class {rdf_array_data_source_class_name} final : public ROOT::RDF::RDataSource {{
private:
    unsigned int fNSlots{{0U}};

    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    std::vector<std::pair<ssize_t, ssize_t*>> fColDataPointers;

    // FIXME: define entry ranges for each column???
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{{ {{0ull, 3ull}} }};

    // type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {{
        cout << "#2. GetColumnReadersImpl for " << colName;
        const auto index = std::distance(fColNames.begin(), std::find(fColNames.begin(), fColNames.end(), colName));
        cout << "index " << index << endl;

        auto colNameStr = std::string(colName);
        const auto idName = ROOT::Internal::RDF::TypeID2TypeName(id);
        cout << " and type " << idName << endl;

        return {{}};
    }}

    size_t GetEntriesNumber() {{
        return fColNames.size();
    }}

public:
    {rdf_array_data_source_class_name}(ColumnTypes&&... wrappers)
        :
        """

        # wrappers.name...
        cpp_code_column_names = """fColNames({""".strip()
        k = 0
        for key in columns:
            cpp_code_column_names = (
                cpp_code_column_names
                + f"""
            "awkward:{key}"
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_column_names = cpp_code_column_names + ", "

        cpp_code_column_names = cpp_code_column_names + "}),"

        # wrappers.type...
        cpp_code_wrappers_type = """fColTypeNames({""".strip()
        k = 0
        for key in columns:
            cpp_code_wrappers_type = (
                cpp_code_wrappers_type
                + f"""
            "{type(rdf_array_view_entries[key]).__cpp_name__}"
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_wrappers_type = cpp_code_wrappers_type + ", "
        cpp_code_wrappers_type = cpp_code_wrappers_type + "}),"

        # column type map
        cpp_code_column_types_map = """fColTypesMap({""".strip()
        k = 0
        for key in columns:
            cpp_code_column_types_map = (
                cpp_code_column_types_map
                + f"""
            {{ "awkward:{key}", "{type(rdf_array_view_entries[key]).__cpp_name__}" }}
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_column_types_map = cpp_code_column_types_map + ", "

        cpp_code_column_types_map = cpp_code_column_types_map + "}),"

        # data pointers
        cpp_code_data_pointers = (
            """fColDataPointers({{wrappers.length, wrappers.ptrs}...})"""
        )

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_column_names
            + cpp_code_wrappers_type
            + cpp_code_column_types_map
            + cpp_code_data_pointers
            + f"""
    {{
        cout << endl << "Construct an {rdf_array_data_source_class_name} with ";
        cout << GetEntriesNumber()  << " columns named:" << endl;

        for (int64_t i = 0; i < fColNames.size(); i++) {{
            cout << fColNames[i] << ", ";
        }}
        cout << endl << "column types:" << endl;
        for (auto t : fColTypeNames) {{
            cout << t << ", ";
        }}
        cout << endl << "mapped:" << endl;
        for (auto it : fColTypesMap) {{
            cout << it.first << ": " << it.second << ", ";
        }}
        cout << endl << "." << endl;
    }}

    ~{rdf_array_data_source_class_name}() {{
        // FIXME: no need for the destructor
    }}

    void SetNSlots(unsigned int nSlots) {{
        cout << "#1. SetNSlots " << nSlots << endl;
        fNSlots = nSlots; // FIXME: always 1 slot for now
    }}

    std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
    GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info & /*tid*/) {{
        cout << "#2.2. GetColumnReaders" << endl;
        const auto index = std::distance(fColNames.begin(), std::find(fColNames.begin(), fColNames.end(), name));
        cout << "index " << index << endl;
        """
        )

        cpp_code_readers = f"""
        // Instantiate an array reader for selected column based on {key}:
        switch (index) {{
        """
        indx = 0
        for key in columns:
            cpp_code_readers = (
                cpp_code_readers
                + f"""
            case {indx}:
                cout << "index is " << {indx} << " and "
                << fColDataPointers[{indx}].first << " length of data at " << fColDataPointers[{indx}].second << endl;
                return std::unique_ptr<awkward::AwkwardArrayColumnReader_{rdf_generated_types[key]}_{key}>(
                    new awkward::AwkwardArrayColumnReader_{rdf_generated_types[key]}_{key}(
                        fColDataPointers[{indx}].first,
                        fColDataPointers[{indx}].second
                    )
                );
            """
            )
            indx = indx + 1

        cpp_code_readers = (
            cpp_code_readers
            + f"""
        // If {key} column reader is not defined:
        default:
            std::string err = "The specified column name, \"" + name + "\" does not have a reader defined.";
            throw std::runtime_error(err);
        }}
        """
        ).strip()

        cpp_code_end = f"""
        // Done generating code for a specific {key} in Python. Proceed with C++:
   }}

    void Initialise() {{
        cout << "#3. Initialise" << endl;
        const auto nEntries = GetEntriesNumber();
        cout << "nEntries " << nEntries << endl;
        const auto nEntriesInRange = nEntries / fNSlots; // always one for now
        cout << "nEntriesInRange " << nEntriesInRange << endl;
        auto reminder = 1U == fNSlots ? 0 : nEntries % fNSlots;
        cout << "reminder " << reminder << endl;
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
    return new ROOT::RDataFrame(std::make_unique<{rdf_array_data_source_class_name}<ColumnTypes...>>(std::move(wrappers)...));
}}
        """

        cpp_code = cpp_code_begin + cpp_code_readers + cpp_code_end
        done = compiler(cpp_code)
        assert done is True

    rdf = ROOT.MakeAwkwardArrayDS(*rdf_list_of_wrappers)

    return (
        rdf,
        rdf_column_readers,
        rdf_lookups,
        rdf_generators,
    )
