# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import ctypes

# C++17 is required for invoke
headers = "functional"
f_cache = {}
f_type = {}
cache = {}


def generate_ArrayBuilder(compiler):
    key = "ArrayBuilderShim"

    out = cache.get(key)

    if out is None:
        out = f"""namespace awkward {{

    class ArrayBuilderShim {{
    public:
        typedef uint8_t (*FuncPtr)(void*);
        typedef uint8_t (*FuncPtr_Int)(void*, int64_t);
        typedef uint8_t (*FuncPtr_Bool)(void*, bool);
        typedef uint8_t (*FuncPtr_Dbl)(void*, double);
        typedef uint8_t (*FuncPtr_CharPtr)(void*, const char*);

        ArrayBuilderShim(void* ptr)
            : ptr_(ptr) {{ }}

        uint8_t
        beginlist() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginlist, ctypes.c_voidp).value})), ptr_);
        }}

        uint8_t
        beginrecord() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord, ctypes.c_voidp).value})), ptr_);
        }}

        uint8_t
        beginrecord_check(const char* name) {{
            return std::invoke(reinterpret_cast<FuncPtr_CharPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_check, ctypes.c_voidp).value})), ptr_, name);
        }}

        uint8_t
        beginrecord_fast(const char* name) {{
            return std::invoke(reinterpret_cast<FuncPtr_CharPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_beginrecord_fast, ctypes.c_voidp).value})), ptr_, name);
        }}

        uint8_t
        begintuple(int64_t numfields) {{
            return std::invoke(reinterpret_cast<FuncPtr_Int>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_begintuple, ctypes.c_voidp).value})), ptr_, numfields);
        }}

        uint8_t
        boolean(bool x) {{
            return std::invoke(reinterpret_cast<FuncPtr_Bool>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_boolean, ctypes.c_void_p).value})), ptr_, x);
        }}

        uint8_t
        clear() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_clear, ctypes.c_void_p).value})), ptr_);
        }}

        uint8_t
        endlist() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endlist, ctypes.c_void_p).value})), ptr_);
        }}

        uint8_t
        endrecord() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endrecord, ctypes.c_void_p).value})), ptr_);
        }}

        uint8_t
        endtuple() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_endtuple, ctypes.c_void_p).value})), ptr_);
        }}

        uint8_t
        field_check(const char* key) {{
            return std::invoke(reinterpret_cast<FuncPtr_CharPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_field_check, ctypes.c_void_p).value})), ptr_, key);
        }}

        uint8_t
        field_fast(const char* key) {{
            return std::invoke(reinterpret_cast<FuncPtr_CharPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_field_fast, ctypes.c_void_p).value})), ptr_, key);
        }}

        uint8_t
        index(int64_t index) {{
            return std::invoke(reinterpret_cast<FuncPtr_Int>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_index, ctypes.c_void_p).value})), ptr_, index);
        }}

        uint8_t
        integer(int64_t x) {{
            return std::invoke(reinterpret_cast<FuncPtr_Int>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_integer, ctypes.c_void_p).value})), ptr_, x);
        }}

        uint8_t
        null() {{
            return std::invoke(reinterpret_cast<FuncPtr>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_null, ctypes.c_void_p).value})), ptr_);
        }}

        uint8_t
        real(double x) {{
            return std::invoke(reinterpret_cast<FuncPtr_Dbl>(reinterpret_cast<long>({ctypes.cast(ak._libawkward.ArrayBuilder_real, ctypes.c_void_p).value})), ptr_, x);
        }}

    private:
        void* ptr_;
}};
}}
""".strip()
        err = compiler(out)
        assert err is True

    return "awkward::" + key


def array_builder(builder):
    import ROOT

    return ROOT.awkward.ArrayBuilderShim(
        ctypes.cast(builder._layout._ptr, ctypes.c_voidp)
    )


def connect_ArrayBuilder(compiler, builder):

    tag = hex(builder._layout._ptr)

    f_cache["beginlist"] = f"beginlist_of_{tag}"
    f_cache["beginrecord"] = f"beginrecord_of_{tag}"
    f_cache["beginrecord_check"] = f"beginrecord_check_of_{tag}"
    f_cache["beginrecord_fast"] = f"beginrecord_fast_of_{tag}"
    f_cache["begintuple"] = f"begintuple_of_{tag}"
    f_cache["boolean"] = f"boolean_of_{tag}"
    f_cache["clear"] = f"clear_of_{tag}"
    f_cache["endlist"] = f"endlist_of_{tag}"
    f_cache["endrecord"] = f"endrecord_of_{tag}"
    f_cache["endtuple"] = f"endtuple_of_{tag}"
    f_cache["field_check"] = f"field_check_of_{tag}"
    f_cache["field_fast"] = f"field_fast_of_{tag}"
    f_cache["index"] = f"index_of_{tag}"
    f_cache["integer"] = f"integer_of_{tag}"
    f_cache["null"] = f"null_of_{tag}"
    f_cache["real"] = f"real_of_{tag}"

    f_type["FuncPtr"] = f"FuncPtr_of_{tag}"
    f_type["FuncPtr_Int"] = f"FuncPtr_Int_of_{tag}"
    f_type["FuncPtr_Bool"] = f"FuncPtr_Bool_of_{tag}"
    f_type["FuncPtr_Dbl"] = f"FuncPtr_Dbl_of_{tag}"
    f_type["FuncPtr_CharPtr"] = f"FuncPtr_CharPtr_of_{tag}"

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


def to_awkward_array(data_frame, columns=None, exclude=None, columns_as_records=True):
    builder = ak.ArrayBuilder()

    def list_of_doubles(data):
        builder.begin_list()
        for x in data:
            builder.real(x)
        builder.end_list()

    def list_of_integers(data):
        builder.begin_list()
        for x in data:
            builder.integer(x)
        builder.end_list()

    def count_dimentions(column_type):
        rvec_count = column_type.count("RVec<")
        v_count = column_type.count("vector<")
        rvec_count = 0 if rvec_count == -1 else rvec_count
        v_count = 0 if v_count == -1 else v_count
        return rvec_count + v_count

    def type_of_nested_data(column_type):
        t1 = column_type.rfind("<")
        t2 = column_type.find(">")
        return column_type if t1 == -1 and t2 == -1 else column_type[t1 + 1 : t2]

    # FIXME:
    def func_real(data, n):
        for x in data:
            _ = builder.real(x) if n <= 0 else func_real(x, n - 1)

    # FIXME:
    def func_integer(data, n):
        for x in data:
            _ = builder.integer(x) if n <= 0 else func_integer(x, n - 1)

    # Find all column names in the dataframe
    if not columns:
        columns = [str(x) for x in data_frame.GetColumnNames()]

    # Exclude the specified columns
    if exclude is None:
        exclude = []
    columns = [x for x in columns if x not in exclude]

    if columns_as_records:
        column_type = {}
        type = {}
        result_ptrs = {}
        for col in columns:
            # FIXME: not stable???
            # `GetColumnType` fails occasionally with a `SystemError`
            column_type[col] = data_frame.GetColumnType(col)
            type[col] = type_of_nested_data(column_type[col])
            result_ptrs[col] = data_frame.Take[column_type[col]](col)

        # separate it from the above for performance reasons: `GetValue()` triggers
        # an RDF event loop.
        cpp_reference = {}
        for col in columns:
            cpp_reference[col] = result_ptrs[col].GetValue()

        for col in columns:
            builder.begin_record(col)
            builder.field(col)

            # check if the column type is either `ROOT::VecOps::RVec` or
            # `std::vector`, possibly nested,
            # for example, `RVec<RVec<int>>`
            result = count_dimentions(column_type[col])

            for _ in range(result):
                builder.begin_list()

            if type[col] == "double":
                _ = (
                    func_real(cpp_reference[col], result)
                    if result > 0
                    else list_of_doubles(cpp_reference[col])
                )
            elif type[col] == "int":
                _ = (
                    func_integer(cpp_reference[col], result)
                    if result > 0
                    else list_of_integers(cpp_reference[col])
                )
            else:
                raise ak._v2._util.error(NotImplementedError)

            for _ in range(result):
                builder.end_list()

            builder.end_record()
    else:
        raise ak._v2._util.error(NotImplementedError)

    return builder.snapshot()
