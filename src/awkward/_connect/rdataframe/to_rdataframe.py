# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import threading

import ROOT

import awkward as ak
import awkward._connect.cling
import awkward._lookup

compiler_lock = threading.Lock()

cache = {}


def compile(source_code):
    with compiler_lock:
        return ROOT.gInterpreter.Declare(source_code)


compile(
    """
#include <Python.h>
"""
)


def to_rdataframe(layouts, length, flatlist_as_rvec):
    return DataSourceGenerator(length, flatlist_as_rvec=flatlist_as_rvec).data_frame(
        layouts
    )


class DataSourceGenerator:
    def __init__(self, length, flatlist_as_rvec=True, use_cached=True):
        self.length = length
        self.flatlist_as_rvec = flatlist_as_rvec
        self.use_cached = use_cached
        self.entry_types = {}
        self.data_ptrs_list = []
        self.generators = {}
        self.lookups = {}

    def class_type(self):

        class_type_suffix = ""
        for key, value in self.generators.items():
            class_type_suffix = class_type_suffix + "_" + key + "_" + value.class_type()

        key = ak._util.identifier_hash(class_type_suffix)

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
            self.generators[key] = ak._connect.cling.togenerator(
                layouts[key].form, flatlist_as_rvec=self.flatlist_as_rvec
            )
            self.lookups[key] = ak._lookup.Lookup(layouts[key], self.generators[key])
            self.generators[key].generate(ROOT.gInterpreter.Declare)

            self.entry_types[key] = self.generators[key].entry_type()

            if isinstance(self.generators[key], ak._connect.cling.NumpyArrayGenerator):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.ListArrayGenerator
            ) and (
                self.generators[key].is_string
                or (
                    self.generators[key].flatlist_as_rvec
                    and self.generators[key].is_flatlist
                )
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.RegularArrayGenerator
            ) and (
                self.generators[key].flatlist_as_rvec
                and self.generators[key].is_flatlist
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.IndexedOptionArrayGenerator
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.IndexedArrayGenerator
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.ByteMaskedArrayGenerator
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.BitMaskedArrayGenerator
            ):
                pass
            elif isinstance(
                self.generators[key], ak._connect.cling.UnmaskedArrayGenerator
            ):
                pass
            else:
                self.entry_types[key] = "awkward::" + self.entry_types[key]

            self.data_ptrs_list.append(self.lookups[key].arrayptrs.ctypes.data)

            if self.entry_types[key] == "bool":
                cpp_code_declare_slots = (
                    cpp_code_declare_slots
                    + f"""
            ULong64_t fPtrs_{key} = 0;
            std::vector<uint8_t>  slots_{key};
            std::vector<uint8_t*> addrs_{key};
        """
                )
            else:
                cpp_code_declare_slots = (
                    cpp_code_declare_slots
                    + f"""
            ULong64_t fPtrs_{key} = 0;
            std::vector<{self.entry_types[key]}>  slots_{key};
            std::vector<{self.entry_types[key]}*> addrs_{key};
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
        "{self.entry_types[key]}"
    """.strip()
            )

            cpp_code_column_types_map = (
                cpp_code_column_types_map
                + f"""
        {{ "{key}", "{self.entry_types[key]}" }}
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
        slots_{key}[slot] = awkward::{self.generators[key].class_type()}(0, fSize, 0, reinterpret_cast<ssize_t*>(fPtrs_{key}), fPyLookup)[entry];
    """
            )

            k = k + 1
            if k < len(layouts):
                cpp_code_column_names = cpp_code_column_names + ", "
                cpp_code_column_type_names = cpp_code_column_type_names + ", "
                cpp_code_column_types_map = cpp_code_column_types_map + ", "

        array_data_source = self.class_type()

        if self.use_cached:
            cpp_code = cache.get(array_data_source)
        else:
            cpp_code = None

        if cpp_code is None:
            cpp_code = f"""
namespace awkward {{

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

        PyObject* fPyLookup;

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
        {array_data_source}(PyObject* lookup, ULong64_t size, std::initializer_list<ULong64_t> ptrs_list)
          : fSize(size),
            fPtrs({{ptrs_list}}),
            fColNames({{{cpp_code_column_names}}}),
            fColTypeNames({{{cpp_code_column_type_names}}}),
            fColTypesMap({{{cpp_code_column_types_map}}}),
            fPyLookup(lookup)
            {{
                Py_INCREF(fPyLookup);
                {cpp_code_init_slots}
            }}

        ~{array_data_source}() {{
            Py_DECREF(fPyLookup);
        }}

        void SetNSlots(unsigned int nSlots) {{
            fNSlots = nSlots;

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

    ROOT::RDataFrame* MakeAwkwardArrayDS_{array_data_source}(PyObject* lookup, ULong64_t size, std::initializer_list<ULong64_t> ptrs_list) {{
        return new ROOT::RDataFrame(std::make_unique<{array_data_source}>(std::forward<PyObject*>(lookup), size, ptrs_list));
    }}

}}
            """
            cache[array_data_source] = cpp_code
            done = compile(cpp_code)
            assert done is True

        rdf = getattr(ROOT.awkward, f"MakeAwkwardArrayDS_{array_data_source}")(
            self.lookups,
            self.length,
            (self.data_ptrs_list),
        )

        rdf = rdf.Define("awkward_index_", "(int64_t)rdfentry_")

        return rdf
