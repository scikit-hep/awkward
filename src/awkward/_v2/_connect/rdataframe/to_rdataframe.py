# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT

compiler = ROOT.gInterpreter.Declare
# def compiler(source_code):
#     print(source_code)
#     return ROOT.gInterpreter.Declare(source_code)


def to_rdataframe(layouts, length, flatlist_as_rvec):
    return DataSourceGenerator(length, flatlist_as_rvec=flatlist_as_rvec).data_frame(
        layouts
    )


class DataSourceGenerator:
    def __init__(self, length, flatlist_as_rvec=False):
        self.length = length
        self.flatlist_as_rvec = flatlist_as_rvec
        self.data_ptrs_list = []
        self.generators = {}
        self.lookups = {}

    def class_type(self):
        key = hash(zip(self.generators.keys(), self.generators.values()))
        return f"AwkwardArrayDataSource_{key}"

    def data_frame(self, layouts):
        cpp_code_declare_slots = ""
        cpp_code_define_readers = ""
        cpp_code_column_names = ""
        cpp_code_column_type_names = ""
        cpp_code_column_types_map = ""
        cpp_code_init_slots = ""
        cpp_code_resize_slots = ""
        cpp_code_entries = ""

        k = 0

        for key in layouts:
            layout = layouts[key]
            self.generators[key] = ak._v2._connect.cling.togenerator(
                layout.form, flatlist_as_rvec=self.flatlist_as_rvec
            )
            self.lookups[key] = ak._v2._lookup.Lookup(layout)
            generator = self.generators[key]
            generator.generate(compiler)

            entry_type = generator.entry_type()
            if isinstance(generator, ak._v2._connect.cling.NumpyArrayGenerator):
                pass
            elif isinstance(generator, ak._v2._connect.cling.ListArrayGenerator) and (
                generator.is_string
                or (generator.flatlist_as_rvec and generator.is_flatlist)
            ):
                pass
            else:
                entry_type = "awkward::" + entry_type

            self.data_ptrs_list.append(self.lookups[key].arrayptrs.ctypes.data)

            cpp_code_declare_slots = (
                cpp_code_declare_slots
                + f"""
        ULong64_t fPtrs_{key} = 0;
        std::vector<{entry_type}>  slots_{key};
        std::vector<{entry_type}*> addrs_{key};
    """
            )

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
            cpp_code_define_readers = cpp_code_define_readers + "else "

            cpp_code_column_names = (
                cpp_code_column_names
                + f"""
        "{key}"
    """.strip()
            )

            cpp_code_column_type_names = (
                cpp_code_column_type_names
                + f"""
        "{entry_type}"
    """.strip()
            )

            cpp_code_column_types_map = (
                cpp_code_column_types_map
                + f"""
        {{ "{key}", "{entry_type}" }}
    """.strip()
            )

            cpp_code_init_slots = (
                cpp_code_init_slots
                + f"""
        fPtrs_{key} = fPtrs[{k}];
    """
            )

            cpp_code_resize_slots = (
                cpp_code_resize_slots
                + f"""
        slots_{key}.resize(fNSlots);
        addrs_{key}.resize(fNSlots);
    """
            )

            cpp_code_entries = (
                cpp_code_entries
                + f"""
        slots_{key}[slot] = awkward::{self.generators[key].class_type()}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_{key}))[entry];
    """
            )

            k = k + 1
            if k < len(layouts):
                cpp_code_column_names = cpp_code_column_names + ", "
                cpp_code_column_type_names = cpp_code_column_type_names + ", "
                cpp_code_column_types_map = cpp_code_column_types_map + ", "

        array_data_source = self.class_type()

        if not hasattr(ROOT, array_data_source):
            cpp_code = f"""
    class {array_data_source} final
      : public ROOT::RDF::RDataSource {{
    private:
        ULong64_t fSize = 0ULL;
        std::vector<ULong64_t> fPtrs;
        unsigned int fNSlots{{0U}};
        const std::vector<std::string> fColNames;
        const std::vector<std::string> fColTypeNames;
        const std::map<std::string, std::string> fColTypesMap;
        std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{{ }};

        {cpp_code_declare_slots}

        Record_t
        GetColumnReadersImpl(std::string_view name, const std::type_info &id) {{
            Record_t reader;

            {cpp_code_define_readers}
            {{
                for (auto i : ROOT::TSeqU(fNSlots)) {{
                    reader.emplace_back(nullptr);
                }}
            }}
            return reader;
        }}

    public:
        {array_data_source}(ULong64_t size, std::initializer_list<ULong64_t> ptrs_list)
          : fSize(size),
            fPtrs({{ptrs_list}}),
            fColNames({{{cpp_code_column_names}}}),
            fColTypeNames({{{cpp_code_column_type_names}}}),
            fColTypesMap({{{cpp_code_column_types_map}}})
            {{
                {cpp_code_init_slots}
            }}

            void SetNSlots(unsigned int nSlots) {{
                fNSlots = nSlots; // FIXME: always 1 slot for now

                {cpp_code_resize_slots}
            }}

        void Initialise() {{
            // initialize fEntryRanges
            const auto chunkSize = fSize / fNSlots;
            auto start = 0UL;
            auto end = 0UL;
            for (auto i : ROOT::TSeqUL(fNSlots)) {{
                start = end;
                end += chunkSize;
                fEntryRanges.emplace_back(start, end);
                (void)i;
             }}
             // TODO: redistribute reminder to all slots
             fEntryRanges.back().second += fSize % fNSlots;
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
            {cpp_code_entries}
            return true;
        }}
    }};

    ROOT::RDataFrame* MakeAwkwardArrayDS_{array_data_source}(ULong64_t size, std::initializer_list<ULong64_t> ptrs_list) {{
        return new ROOT::RDataFrame(std::make_unique<{array_data_source}>(size, ptrs_list));
    }}
            """

            done = compiler(cpp_code)
            assert done is True

        rdf = getattr(ROOT, f"MakeAwkwardArrayDS_{array_data_source}")(
            self.length,
            (self.data_ptrs_list),
        )

        return rdf
