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
        struct ArrayWrapper {
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
            ptrs(ptrs) { }

            const std::string_view name;
            const std::string_view type;
            const ssize_t length;
            ssize_t* ptrs;
        };
    }
        """
        )
        assert done is True

    rdf_list_of_wrappers = []
    rdf_generated_types = {}
    rdf_array_view_entries = {}

    rdf_array_data_source_class_name = "AwkwardArrayDataSource_of"

    for key in columns:
        layout = columns[key].layout
        generator = ak._v2._connect.cling.togenerator(layout.form)
        lookup = ak._v2._lookup.Lookup(layout)

        generator.generate(compiler, flatlist_as_rvec=flatlist_as_rvec)
        generated_type = generator.entry_type()
        rdf_generated_types[key] = generated_type

        # Generate a unique class name
        rdf_array_data_source_class_name = (
            rdf_array_data_source_class_name + f"""_{generated_type}_as_{key}"""
        )

        if not hasattr(ROOT, f"get_entry_{generated_type}_{key}_{flatlist_as_rvec}"):
            done = compiler(
                f"""
            auto get_entry_{generated_type}_{key}_{flatlist_as_rvec}(ssize_t length, ssize_t* ptrs, int64_t i) {{
                return {generator.entry(flatlist_as_rvec=flatlist_as_rvec)};
            }}
            """.strip()
            )
            assert done is True

            rdf_array_view_entries[key] = getattr(
                ROOT, f"get_entry_{generated_type}_{key}_{flatlist_as_rvec}"
            )(len(layout), lookup.arrayptrs, 0)

        array_wrapper = ROOT.awkward.ArrayWrapper(
            f"awkward:{key}",
            type(rdf_array_view_entries[key]).__cpp_name__,
            len(layout),
            lookup.arrayptrs,
        )

        if not hasattr(
            ROOT,
            f"awkward::AwkwardArrayColumnReader_{generated_type}_{key}_{flatlist_as_rvec}",
        ):
            done = compiler(
                f"""
namespace awkward {{
    class AwkwardArrayColumnReader_{generated_type}_{key}_{flatlist_as_rvec} : public ROOT::Detail::RDF::RColumnReaderBase {{
    public:
        AwkwardArrayColumnReader_{generated_type}_{key}_{flatlist_as_rvec}(ssize_t length, ssize_t* ptrs)
            : length(length),
              ptrs(ptrs),
              view_(get_entry_{generated_type}_{key}_{flatlist_as_rvec}(length, ptrs, 0)) {{ }}

        ssize_t length;
        ssize_t* ptrs;

    private:
        void* GetImpl(Long64_t entry) {{
            view_ = get_entry_{generated_type}_{key}_{flatlist_as_rvec}(length, ptrs, entry);
            return reinterpret_cast<void*>(&view_);
        }}

        {type(rdf_array_view_entries[key]).__cpp_name__} view_;
    }};
}}
    """.strip()
            )
            assert done is True

        rdf_list_of_wrappers.append(array_wrapper)

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
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{{ }};

    // type-erased vector of pointers to pointers to column values - one per slot
    Record_t
    GetColumnReadersImpl(std::string_view colName, const std::type_info &id) {{
        return {{ }};
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
            + """
    { }

    void SetNSlots(unsigned int nSlots) {
        fNSlots = nSlots; // FIXME: always 1 slot for now
    }

    std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
    GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info & /*tid*/) {
        const auto index = std::distance(fColNames.begin(), std::find(fColNames.begin(), fColNames.end(), name));
        """
        )

        cpp_code_readers = """
        switch (index) {
        """
        indx = 0
        for key in columns:
            cpp_code_readers = (
                cpp_code_readers
                + f"""
            case {indx}:
                return std::unique_ptr<awkward::AwkwardArrayColumnReader_{rdf_generated_types[key]}_{key}_{flatlist_as_rvec}>(
                    new awkward::AwkwardArrayColumnReader_{rdf_generated_types[key]}_{key}_{flatlist_as_rvec}(
                        fColDataPointers[{indx}].first,
                        fColDataPointers[{indx}].second
                    )
                );
            """
            )
            indx = indx + 1

        cpp_code_readers = (
            cpp_code_readers
            + """
        default:
            std::string err = "The specified column name, \"" + name + "\" does not have a reader defined.";
            throw std::runtime_error(err);
        }
        """
        ).strip()

        cpp_code_end = f"""
   }}

    void Initialise() {{
        fEntryRanges.resize(1);
        fEntryRanges[0].first = 0ull;
        fEntryRanges[0].second = fColDataPointers[0].first; // FIXME: the range is defined by the first column?
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
        auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
        return entryRanges;
    }}

    bool SetEntry(unsigned int slot, ULong64_t entry) {{
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

    return (rdf,)
