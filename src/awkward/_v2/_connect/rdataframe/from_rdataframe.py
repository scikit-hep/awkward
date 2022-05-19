# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

cppyy.add_include_path("include")

compiler = ROOT.gInterpreter.Declare
numpy = ak.nplike.Numpy.instance()

# C++17 is required for invoke
headers = "functional"
f_cache = {}
f_type = {}


def connect_ArrayBuilder(compiler, builder):
    import ctypes

    f_cache["beginlist"] = f"beginlist_of_{builder._layout._ptr}"
    f_cache["beginrecord"] = f"beginrecord_of_{builder._layout._ptr}"
    f_cache["beginrecord_check"] = f"beginrecord_check_of_{builder._layout._ptr}"
    f_cache["beginrecord_fast"] = f"beginrecord_fast_of_{builder._layout._ptr}"
    f_cache["begintuple"] = f"begintuple_of_{builder._layout._ptr}"
    f_cache["boolean"] = f"boolean_of_{builder._layout._ptr}"
    f_cache["clear"] = f"clear_of_{builder._layout._ptr}"
    f_cache["endlist"] = f"endlist_of_{builder._layout._ptr}"
    f_cache["endrecord"] = f"endrecord_of_{builder._layout._ptr}"
    f_cache["endtuple"] = f"endtuple_of_{builder._layout._ptr}"
    f_cache["field_check"] = f"field_check_of_{builder._layout._ptr}"
    f_cache["field_fast"] = f"field_fast_of_{builder._layout._ptr}"
    f_cache["index"] = f"index_of_{builder._layout._ptr}"
    f_cache["integer"] = f"integer_of_{builder._layout._ptr}"
    f_cache["null"] = f"null_of_{builder._layout._ptr}"
    f_cache["real"] = f"real_of_{builder._layout._ptr}"

    f_type["FuncPtr"] = f"FuncPtr_of_{builder._layout._ptr}"
    f_type["FuncPtr_Int"] = f"FuncPtr_Int_of_{builder._layout._ptr}"
    f_type["FuncPtr_Bool"] = f"FuncPtr_Bool_of_{builder._layout._ptr}"
    f_type["FuncPtr_Dbl"] = f"FuncPtr_Dbl_of_{builder._layout._ptr}"
    f_type["FuncPtr_CharPtr"] = f"FuncPtr_CharPtr_of_{builder._layout._ptr}"

    out = f"""
    typedef uint8_t (*{f_type["FuncPtr"]})(void*);
    typedef uint8_t (*{f_type["FuncPtr_Int"]})(void*, int64_t);
    typedef uint8_t (*{f_type["FuncPtr_Bool"]})(void*, bool);
    typedef uint8_t (*{f_type["FuncPtr_Dbl"]})(void*, double);
    typedef uint8_t (*{f_type["FuncPtr_CharPtr"]})(void*, const char*);

    uint8_t
    {f_cache["beginlist"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["beginrecord"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["beginrecord_check"]}(const char* name) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_CharPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_check, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}), name);
    }}

    uint8_t
    {f_cache["beginrecord_fast"]}(const char* name) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_CharPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_fast, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}), name);
    }}

    uint8_t
    {f_cache["begintuple"]}(int64_t numfields) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_Int"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_begintuple, ctypes.c_voidp).value})), reinterpret_cast<void *>({builder._layout._ptr}), numfields);
    }}

    uint8_t
    {f_cache["boolean"]}(bool x) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_Bool"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_boolean, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), x);
    }}

    uint8_t
    {f_cache["clear"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_clear, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["endlist"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endlist, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["endrecord"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endrecord, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["endtuple"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endtuple, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["field_check"]}(const char* key) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_CharPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_field_check, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), key);
    }}

    uint8_t
    {f_cache["field_fast"]}(const char* key) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_CharPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_field_fast, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), key);
    }}

    uint8_t
    {f_cache["index"]}(int64_t index) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_Int"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_index, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), index);
    }}

    uint8_t
    {f_cache["integer"]}(int64_t x) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_Int"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_integer, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), x);
    }}

    uint8_t
    {f_cache["null"]}() {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_null, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}));
    }}

    uint8_t
    {f_cache["real"]}(double x) {{
        return std::invoke(reinterpret_cast<{f_type["FuncPtr_Dbl"]}>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_real, ctypes.c_void_p).value})), reinterpret_cast<void *>({builder._layout._ptr}), x);
    }}
    """.strip()
    compiler(out)

    return f_cache


compiler(
    """
#include <iterator>
#include <stdlib.h>

extern "C" {
  /// @brief C interface to {@link awkward::ArrayBuilder#null ArrayBuilder::null}.
   uint8_t
    awkward_ArrayBuilder_null(void* arraybuilder);

  /// @brief C interface to {@link awkward::ArrayBuilder#boolean ArrayBuilder::boolean}.
   uint8_t
    awkward_ArrayBuilder_boolean(void* arraybuilder,
                                 bool x);

  /// @brief C interface to {@link awkward::ArrayBuilder#integer ArrayBuilder::integer}.
   uint8_t
    awkward_ArrayBuilder_integer(void* arraybuilder,
                                 int64_t x);

  /// @brief C interface to {@link awkward::ArrayBuilder#real ArrayBuilder::real}.
   uint8_t
    awkward_ArrayBuilder_real(void* arraybuilder,
                              double x);

  /// @brief C interface to {@link awkward::ArrayBuilder#complex ArrayBuilder::complex}.
   uint8_t
    awkward_ArrayBuilder_complex(void* arraybuilder,
                                 double real,
                                 double imag);

  /// @brief C interface to {@link awkward::ArrayBuilder#datetime ArrayBuilder::datetime}.
   uint8_t
    awkward_ArrayBuilder_datetime(void* arraybuilder,
                                  int64_t x,
                                  const char* unit);

  /// @brief C interface to {@link awkward::ArrayBuilder#timedelta ArrayBuilder::timedelta}.
   uint8_t
    awkward_ArrayBuilder_timedelta(void* arraybuilder,
                                   int64_t x,
                                   const char* unit);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#bytestring ArrayBuilder::bytestring}.
   uint8_t
    awkward_ArrayBuilder_bytestring(void* arraybuilder,
                                    const char* x);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#bytestring ArrayBuilder::bytestring}.
   uint8_t
    awkward_ArrayBuilder_bytestring_length(void* arraybuilder,
                                           const char* x,
                                           int64_t length);

  /// @brief C interface to {@link awkward::ArrayBuilder#string ArrayBuilder::string}.
   uint8_t
    awkward_ArrayBuilder_string(void* arraybuilder,
                                const char* x);

  /// @brief C interface to {@link awkward::ArrayBuilder#string ArrayBuilder::string}.
   uint8_t
    awkward_ArrayBuilder_string_length(void* arraybuilder,
                                       const char* x,
                                       int64_t length);

  /// @brief C interface to
  /// {@link awkward::ArrayBuilder#beginlist ArrayBuilder::beginlist}.
   uint8_t
    awkward_ArrayBuilder_beginlist(void* arraybuilder);

  /// @brief C interface to {@link awkward::ArrayBuilder#endlist ArrayBuilder::endlist}.
   uint8_t
    awkward_ArrayBuilder_endlist(void* arraybuilder);

}

namespace awkward {

template<typename T>
T* copy_ptr(const T* from_ptr, int64_t size)
{
    T* array = malloc(sizeof(T)*size);
    for (int64_t i = 0; i < size; i++) {
        array[i] = from_ptr[i];
    }
    return array;
}

template <typename T>
void
fill_real(void* ptr, const T& data) {
    typedef typename T::value_type value_type;
    for (auto const& it: data) {
        awkward_ArrayBuilder_real(ptr, it);
    }
}

template <typename T, typename V>
struct build_array_impl {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
        typedef typename T::value_type value_type;

        cout << "FIXME: processing an iterable of a " << typeid(value_type).name()
            << " type is not implemented yet." << endl;
    };
};

template <typename T>
struct build_array_impl<T, double> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        for (auto const& data: result) {
            awkward_ArrayBuilder_beginlist(ptr);
            for (auto const& it: data) {
                awkward_ArrayBuilder_real(ptr, it);
            }
            awkward_ArrayBuilder_endlist(ptr);
        }
    };
};

template <typename T>
struct build_array_impl<T, int64_t> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
    for (auto const& data: result) {
        awkward_ArrayBuilder_beginlist(ptr);
        for (auto const& it: data) {
            awkward_ArrayBuilder_integer(ptr, it);
        }
        awkward_ArrayBuilder_endlist(ptr);
    }
    };
};

template <typename T>
struct build_array_impl<T, bool> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        for (auto const& data: result) {
            awkward_ArrayBuilder_beginlist(ptr);
            for (auto const& it: data) {
                awkward_ArrayBuilder_boolean(ptr, it);
            }
            awkward_ArrayBuilder_endlist(ptr);
        }
    };
};

template <typename T>
void
build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<void *>(builder_ptr);
    build_array_impl<T, typename T::value_type>::build_array(result, ptr);
}

template <typename T>
std::pair<std::vector<int64_t>, T>
offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result) {

    typedef typename T::value_type value_type;
    std::vector<value_type> data;

    std::vector<int64_t> offsets;
    offsets.reserve(result->size() + 1);
    offsets.emplace_back(0);
    int64_t length = 0;
    std::for_each(result->begin(), result->end(), [&] (auto const& n) {
        length += n.size();
        offsets.emplace_back(length);
        data.insert(data.end(), n.begin(), n.end());
    });

    return {offsets, data};
}

template <typename, typename = void>
constexpr bool is_iterable{};

template <typename T>
constexpr bool is_iterable<
    T,
    std::void_t< decltype(std::declval<T>().begin()),
                 decltype(std::declval<T>().end())
    >
> = true;

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {
};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {
};

template <typename T, typename std::enable_if<is_specialization<T, std::complex>::value, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
    return std::string("complex");
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
    return std::string("primitive");
}

template <typename T, typename std::enable_if<is_iterable<T>, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
    auto str = std::string(typeid(T).name());
    if (str.find("awkward") != string::npos) {
        return std::string("awkward");
    }
    else {
        typedef typename T::value_type value_type;
        if (is_iterable<value_type>) {
            cout << "FIXME: Fast copy is not implemented yet." << endl;
        } else if (std::is_arithmetic<value_type>::value) {
            // build_array(result, builder_ptr);
            return std::string("iterable"); //, offsets_and_flatten(result)};
        }
        return std::string("iterable");
    }
    return "undefined";
}
}
"""
)


def from_rdataframe(data_frame, column, column_as_record=True):
    def _wrap_as_array(column, array, column_as_record):
        return (
            ak._v2.highlevel.Array({column: array})
            if column_as_record
            else ak._v2.highlevel.Array(array)
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)

    ptrs_type = ROOT.awkward.check_type_of[column_type](result_ptrs)

    if ptrs_type in ("primitive", "complex"):
        print("primitive", "complex")

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        content = ak._v2.contents.NumpyArray(numpy.asarray(cpp_reference))

        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[content],
                ),
                highlevel=True,
            )
            if column_as_record
            else ak._v2._util.wrap(content, highlevel=True)
        )

    elif ptrs_type == "iterable":

        builder = ak._v2.highlevel.ArrayBuilder()
        ROOT.awkward.build_array[column_type](result_ptrs, builder._layout._ptr)
        array = builder.snapshot()

        #
        # data_pair = ROOT.offsets_and_flatten[column_type](result_ptrs)
        #
        # content = ak._v2.contents.ListOffsetArray(
        #     ak._v2.index.Index64(data_pair.first),
        #     ak._v2.contents.NumpyArray(numpy.asarray(data_pair.second)),
        # )

        return (
            ak._v2._util.wrap(
                ak._v2.contents.RecordArray(
                    fields=[column],
                    contents=[array.layout],
                ),
                highlevel=True,
            )
            if column_as_record
            else array
            # ak._v2._util.wrap(
            #     content,
            #     highlevel=True,
            # )
        )

    elif ptrs_type == "awkward":
        print("awkward")
        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, cpp_reference, column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
