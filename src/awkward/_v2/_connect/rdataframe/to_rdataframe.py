# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare


def to_rdataframe(columns, flatlist_as_rvec):
    if not columns:
        ak._v2._util.error(
            ValueError("dict of columns must have at least one ak.Array")
        )

    length = len(next(iter(columns.values())))
    for key in columns:
        if len(columns[key]) != length:
            ak._v2._util.error(ValueError("all arrays must be equal length"))

    data_ptrs_list = []
    rdf_generators = {}
    entry_types = {}
    rdf_lookups = {}

    for key in columns:
        layout = columns[key].layout
        rdf_generators[key] = ak._v2._connect.cling.togenerator(layout.form)
        rdf_lookups[key] = ak._v2._lookup.Lookup(layout)
        rdf_generators[key].generate(compiler, flatlist_as_rvec=flatlist_as_rvec)

        entry_types[key] = (
            rdf_generators[key].entry_type(flatlist_as_rvec=flatlist_as_rvec)
            if isinstance(
                rdf_generators[key], ak._v2._connect.cling.NumpyArrayGenerator
            )
            else f"awkward::{rdf_generators[key].entry_type(flatlist_as_rvec=flatlist_as_rvec)}"
        )

        data_ptrs_list.append(rdf_lookups[key].arrayptrs.ctypes.data)

    hashed = hash(zip(rdf_generators.keys(), rdf_generators.values()))
    array_data_source_class_name = f"AwkwardArrayDataSource_{hashed}"

    if not hasattr(ROOT, array_data_source_class_name):
        cpp_code_begin = f"""
class {array_data_source_class_name} final : public ROOT::RDF::RDataSource {{
private:
    ULong64_t fSize = 0ULL;
    std::vector<ULong64_t> fPtrs;
    unsigned int fNSlots{{0U}};

    const std::vector<std::string> fColNames;
    const std::vector<std::string> fColTypeNames;
    const std::map<std::string, std::string> fColTypesMap;
    std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{{ }};
    """

        cpp_code_vectors = """

    """
        for key in columns:
            cpp_code_vectors = (
                cpp_code_vectors
                + f"""
    ULong64_t fPtrs_{key} = 0;
    std::vector<{entry_types[key]}>  slots_{key};
    std::vector<{entry_types[key]}*> addrs_{key};
    """
            )

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_vectors
            + """

    Record_t
    GetColumnReadersImpl(std::string_view name, const std::type_info &id) {
        Record_t reader;
    """
        )

        cpp_code_define_readers = """

    """

        for key in columns:
            cpp_code_define_readers = (
                cpp_code_define_readers
                + f"""
    if (name == "{key}") {{
       for (auto i : ROOT::TSeqU(fNSlots)) {{
          addrs_{key}[i] = &slots_{key}[i];
          reader.emplace_back((void *)(&addrs_{key}[i]));
       }}
    }}
    """
            )

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_define_readers
            + f"""

        return reader;
    }}

public:
    {array_data_source_class_name}(ULong64_t size, std::initializer_list<ULong64_t> ptrs_list)
        : fSize(size),
          fPtrs({{ptrs_list}}),
        """
        )

        cpp_code_column_names = """fColNames({""".strip()
        k = 0
        for key in columns:
            cpp_code_column_names = (
                cpp_code_column_names
                + f"""
            "{key}"
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_column_names = cpp_code_column_names + ", "

        cpp_code_column_names = cpp_code_column_names + "}),"

        cpp_code_wrappers_type = """fColTypeNames({""".strip()
        k = 0
        for key in columns:
            cpp_code_wrappers_type = (
                cpp_code_wrappers_type
                + f"""
            "{entry_types[key]}"
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_wrappers_type = cpp_code_wrappers_type + ", "
        cpp_code_wrappers_type = cpp_code_wrappers_type + "}),"

        cpp_code_column_types_map = """fColTypesMap({""".strip()
        k = 0
        for key in columns:
            cpp_code_column_types_map = (
                cpp_code_column_types_map
                + f"""
            {{ "{key}", "{entry_types[key]}" }}
            """.strip()
            )
            k = k + 1
            if k < len(columns):
                cpp_code_column_types_map = cpp_code_column_types_map + ", "

        cpp_code_column_types_map = cpp_code_column_types_map + "})"

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_column_names
            + cpp_code_wrappers_type
            + cpp_code_column_types_map
            + """
    {
    """
        )

        cpp_code_init_vectors = """

    """
        k = 0
        for key in columns:
            cpp_code_init_vectors = (
                cpp_code_init_vectors
                + f"""
    fPtrs_{key} = fPtrs[{k}];
    """
            )
            k = k + 1

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_init_vectors
            + """
    }

    void SetNSlots(unsigned int nSlots) {
        fNSlots = nSlots; // FIXME: always 1 slot for now
    """
        )

        cpp_code_resize_vectors = """

    """
        for key in columns:
            cpp_code_resize_vectors = (
                cpp_code_resize_vectors
                + f"""
        slots_{key}.resize(fNSlots);
        addrs_{key}.resize(fNSlots);
    """
            )

        cpp_code_begin = (
            cpp_code_begin
            + cpp_code_resize_vectors
            + """
    }

    void Initialise() {
        // initialize fEntryRanges
        const auto chunkSize = fSize / fNSlots;
        auto start = 0UL;
        auto end = 0UL;
        for (auto i : ROOT::TSeqUL(fNSlots)) {
            start = end;
            end += chunkSize;
            fEntryRanges.emplace_back(start, end);
            (void)i;
         }
         // TODO: redistribute reminder to all slots
         fEntryRanges.back().second += fSize % fNSlots;
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

    """
        )
        cpp_code_entries = """

    """
        for key in columns:
            cpp_code_entries = (
                cpp_code_entries
                + f"""
        slots_{key}[slot] = awkward::{rdf_generators[key].class_type((flatlist_as_rvec,))}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_{key}))[entry];
    """
            )

        cpp_code_end = (
            cpp_code_entries
            + f"""
        return true;
    }}
}};

ROOT::RDataFrame* MakeAwkwardArrayDS_{array_data_source_class_name}(ULong64_t size, std::initializer_list<ULong64_t> ptrs_list) {{
    return new ROOT::RDataFrame(std::make_unique<{array_data_source_class_name}>(size, ptrs_list));
}}
        """
        )

        cpp_code = cpp_code_begin + cpp_code_end
        done = compiler(cpp_code)
        assert done is True

    rdf = getattr(ROOT, f"MakeAwkwardArrayDS_{array_data_source_class_name}")(
        length,
        (data_ptrs_list),
    )

    return (rdf, rdf_lookups, rdf_generators)
