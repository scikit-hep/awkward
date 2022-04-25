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

    return DataSourceGenerator(length, flatlist_as_rvec=flatlist_as_rvec).data_frame(
        columns
    )


class DataSourceGenerator:
    def __init__(self, length, flatlist_as_rvec=False):
        self.length = length
        self.flatlist_as_rvec = flatlist_as_rvec
        self.data_ptrs_list = []
        self.generators = {}
        self.lookups = {}
        self.entry_types = {}

    def class_type(self):
        key = hash(zip(self.generators.keys(), self.generators.values()))
        return f"AwkwardArrayDataSource_{key}"

    def data_frame(self, columns):
        for key in columns:
            layout = columns[key].layout
            self.generators[key] = ak._v2._connect.cling.togenerator(layout.form)
            self.lookups[key] = ak._v2._lookup.Lookup(layout)
            self.generators[key].generate(
                compiler, flatlist_as_rvec=self.flatlist_as_rvec
            )

            self.entry_types[key] = (
                self.generators[key].entry_type()
                if isinstance(
                    self.generators[key], ak._v2._connect.cling.NumpyArrayGenerator
                )
                else f"awkward::{self.generators[key].entry_type()}"
            )

            self.data_ptrs_list.append(self.lookups[key].arrayptrs.ctypes.data)

        array_data_source = self.class_type()

        if not hasattr(ROOT, array_data_source):
            cpp_code = (
                f"  class {array_data_source} final"
                "       : public ROOT::RDF::RDataSource {"
                "   private:                                                            "
                "       ULong64_t fSize = 0ULL;"
                "       std::vector<ULong64_t> fPtrs;"
                "       unsigned int fNSlots{0U};"
                "       const std::vector<std::string> fColNames;"
                "       const std::vector<std::string> fColTypeNames;"
                "       const std::map<std::string, std::string> fColTypesMap;"
                "       std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{};"
            )

            cpp_code_declare_slots = """
        """
            for key in columns:
                cpp_code_declare_slots = (
                    cpp_code_declare_slots
                    + f"""
        ULong64_t fPtrs_{key} = 0;
        std::vector<{self.entry_types[key]}>  slots_{key};
        std::vector<{self.entry_types[key]}*> addrs_{key};
        """
                )

            cpp_code = (
                cpp_code
                + cpp_code_declare_slots
                + """
        Record_t
        GetColumnReadersImpl(std::string_view name, const std::type_info &id) {
            Record_t reader;
        """
            )

            cpp_code_define_readers = """
        """
            k = 0
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
                cpp_code_define_readers = cpp_code_define_readers + "else "

            cpp_code = (
                cpp_code
                + cpp_code_define_readers
                + f""" {{
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

            cpp_code_column_type_names = """fColTypeNames({""".strip()
            k = 0
            for key in columns:
                cpp_code_column_type_names = (
                    cpp_code_column_type_names
                    + f"""
                "{self.entry_types[key]}"
                """.strip()
                )
                k = k + 1
                if k < len(columns):
                    cpp_code_column_type_names = cpp_code_column_type_names + ", "
            cpp_code_column_type_names = cpp_code_column_type_names + "}),"

            cpp_code_column_types_map = """fColTypesMap({""".strip()
            k = 0
            for key in columns:
                cpp_code_column_types_map = (
                    cpp_code_column_types_map
                    + f"""
                {{ "{key}", "{self.entry_types[key]}" }}
                """.strip()
                )
                k = k + 1
                if k < len(columns):
                    cpp_code_column_types_map = cpp_code_column_types_map + ", "

            cpp_code_column_types_map = cpp_code_column_types_map + "})"

            cpp_code = (
                cpp_code
                + cpp_code_column_names
                + cpp_code_column_type_names
                + cpp_code_column_types_map
            )

            cpp_code_init_slots = """
        {
        """
            k = 0
            for key in columns:
                cpp_code_init_slots = (
                    cpp_code_init_slots
                    + f"""
        fPtrs_{key} = fPtrs[{k}];
        """
                )
                k = k + 1

            cpp_code = (
                cpp_code
                + cpp_code_init_slots
                + """
        }

        void SetNSlots(unsigned int nSlots) {
            fNSlots = nSlots; // FIXME: always 1 slot for now
        """
            )

            cpp_code_resize_slots = """

        """
            for key in columns:
                cpp_code_resize_slots = (
                    cpp_code_resize_slots
                    + f"""
            slots_{key}.resize(fNSlots);
            addrs_{key}.resize(fNSlots);
        """
                )

            cpp_code = (
                cpp_code
                + cpp_code_resize_slots
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
            slots_{key}[slot] = awkward::{self.generators[key].class_type()}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_{key}))[entry];
        """
                )

            cpp_code = (
                cpp_code
                + cpp_code_entries
                + f"""
            return true;
        }}
    }};

    ROOT::RDataFrame* MakeAwkwardArrayDS_{array_data_source}(ULong64_t size, std::initializer_list<ULong64_t> ptrs_list) {{
        return new ROOT::RDataFrame(std::make_unique<{array_data_source}>(size, ptrs_list));
    }}
            """
            )

            done = compiler(cpp_code)
            assert done is True

        rdf = getattr(ROOT, f"MakeAwkwardArrayDS_{array_data_source}")(
            self.length,
            (self.data_ptrs_list),
        )

        return (rdf, self.lookups, self.generators)
