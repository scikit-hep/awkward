# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import threading

import awkward as ak

_has_checked_version = False
_is_registered = False

_registry = {}

cache = {}

compiler_lock = threading.Lock()


def register_and_check(array=None, cpp_type=None):
    global _has_checked_version

    try:
        import cppyy
    except ImportError as err:
        raise ImportError(
            """install the 'cppyy' package with:

pip install cppyy

or

conda install -c conda-forge cppyy"""
        ) from err

    if not _has_checked_version:
        if ak._util.parse_version(cppyy.__version__) < ak._util.parse_version("2.4.2"):
            raise ImportError(
                "Awkward Array can only work with cppyy 2.4.2 or later "
                "(you have version {})".format(cppyy.__version__)
            )
        _has_checked_version = True

    if array is not None and cpp_type is not None:
        _register(array=array, cpp_type=cpp_type)


def _register(array=None, cpp_type=None):
    import cppyy

    def compile(source_code):
        with compiler_lock:
            done = cppyy.cppdef(source_code)
            return done

    c = ak.cppyy  # noqa: F841

    cpp_code = cache.get(cpp_type)

    if cpp_code is None:
        cpp_code = f"""
    #include "CPyCppyy/API.h"
    class ArrayView_{cpp_type} {{
        int fFlags;
    public:
        ArrayView_{cpp_type}() : fFlags(0) {{}}
        virtual ~ArrayView_{cpp_type}() {{}}
        void setSetArgCalled()     {{ fFlags |= 0x01; }}
        bool wasSetArgCalled()     {{ return fFlags & 0x01; }}
        void setFromMemoryCalled() {{ fFlags |= 0x02; }}
        bool wasFromMemoryCalled() {{ return fFlags & 0x02; }}
        void setToMemoryCalled()   {{ fFlags |= 0x04; }}
        bool wasToMemoryCalled()   {{ return fFlags & 0x04; }}
    }};
    class ArrayView_{cpp_type}Converter : public CPyCppyy::Converter {{
    public:
        virtual bool SetArg(PyObject* pyobject, CPyCppyy::Parameter& para, CPyCppyy::CallContext* = nullptr) {{
            ArrayView_{cpp_type}* a3 = (ArrayView_{cpp_type}*)CPyCppyy::Instance_AsVoidPtr(pyobject);
            a3->setSetArgCalled();
            para.fValue.fVoidp = a3;
            para.fTypeCode = 'V';
            return true;
        }}
        virtual PyObject* FromMemory(void* address) {{
            ArrayView_{cpp_type}* a3 = (ArrayView_{cpp_type}*)address;
            a3->setFromMemoryCalled();
            return CPyCppyy::Instance_FromVoidPtr(a3, "ArrayView_{cpp_type}");
        }}
        virtual bool ToMemory(PyObject* value, void* address) {{
            ArrayView_{cpp_type}* a3 = (ArrayView_{cpp_type}*)address;
            a3->setToMemoryCalled();
            *a3 = *(ArrayView_{cpp_type}*)CPyCppyy::Instance_AsVoidPtr(value);
            return true;
        }}
    }};
    typedef CPyCppyy::ConverterFactory_t cf_t;
    void register_array_view() {{
        std::cout << "Register a3!" << std::endl;
        CPyCppyy::RegisterConverter("ArrayView_{cpp_type}",  (cf_t)+[](CPyCppyy::cdims_t) {{ static ArrayView_{cpp_type}Converter c{{}}; return &c; }});
        CPyCppyy::RegisterConverter("ArrayView_{cpp_type}&", (cf_t)+[](CPyCppyy::cdims_t) {{ static ArrayView_{cpp_type}Converter c{{}}; return &c; }});
    }}
    void unregister_array_view() {{
        CPyCppyy::UnregisterConverter("ArrayView_{cpp_type}");
        CPyCppyy::UnregisterConverter("ArrayView_{cpp_type}&");
    }}
    ArrayView_{cpp_type} gA3a, gA3b;
    void CallWithAPICheck(ArrayView_{cpp_type}&) {{}}
    """.strip()

        cache[cpp_type] = cpp_code
        done = compile(cpp_code)
        assert done is True

    # It is safe to call multiple times: if the C++ type is already registered
    # with cppyy the function will do nothing.
    cppyy.gbl.register_array_view()

    gA3a = cppyy.gbl.gA3a
    assert gA3a
    t = getattr(cppyy.gbl, f"ArrayView_{cpp_type}")
    assert type(gA3a) == t
    assert gA3a.wasFromMemoryCalled()

    # FIXME:assert not gA3a.wasSetArgCalled()
    cppyy.gbl.CallWithAPICheck(gA3a)
    assert gA3a.wasSetArgCalled()

    cppyy.gbl.unregister_array_view()

    gA3b = cppyy.gbl.gA3b
    assert gA3b
    assert type(gA3b) == t
    assert not gA3b.wasFromMemoryCalled()

    return

    if cpp_type is not None:
        # print(f"Register an ArrayView {cpp_type} type converter")
        """Register an ArrayView type converter"""

        cppyy.cppdef(
            f"""
        #include "CPyCppyy/API.h"
        // FIXME: this is moved to ArrayView base class
        // Generate on demand?
        //
        // class {cpp_type} {{
        //     int fFlags;
        // public:
        //     {cpp_type}() : fFlags(0) {{}}
        //     virtual ~{cpp_type}() {{}}
        //     void setSetArgCalled()     {{ fFlags |= 0x01; }}
        //     bool wasSetArgCalled()     {{ return fFlags & 0x01; }}
        //     void setFromMemoryCalled() {{ fFlags |= 0x02; }}
        //     bool wasFromMemoryCalled() {{ return fFlags & 0x02; }}
        //     void setToMemoryCalled()   {{ fFlags |= 0x04; }}
        //     bool wasToMemoryCalled()   {{ return fFlags & 0x04; }}
        // }};
        class {cpp_type}Converter : public CPyCppyy::Converter {{
        public:
            virtual bool SetArg(PyObject* pyobject, CPyCppyy::Parameter& para, CPyCppyy::CallContext* = nullptr) {{
                awkward::{cpp_type}* ak_array_view = (awkward::{cpp_type}*)CPyCppyy::Instance_AsVoidPtr(pyobject);
                ak_array_view->setSetArgCalled();
                para.fValue.fVoidp = ak_array_view;
                para.fTypeCode = 'V';
                return true;
            }}
            virtual PyObject* FromMemory(void* address) {{
                awkward::{cpp_type}* ak_array_view = (awkward::{cpp_type}*)address;
                ak_array_view->setFromMemoryCalled();
                return CPyCppyy::Instance_FromVoidPtr(ak_array_view, "{cpp_type}");
            }}
            virtual bool ToMemory(PyObject* value, void* address) {{
                awkward::{cpp_type}* ak_array_view = (awkward::{cpp_type}*)address;
                ak_array_view->setToMemoryCalled();
                *ak_array_view = *(awkward::{cpp_type}*)CPyCppyy::Instance_AsVoidPtr(value);
                return true;
            }}
        }};

        typedef CPyCppyy::ConverterFactory_t cf_t;

        void register_ak_array_view() {{
            std::cout << "Register awkward::{cpp_type}" << std::endl;

            CPyCppyy::RegisterConverter(\"awkward::{cpp_type}\",  (cf_t)+[](CPyCppyy::cdims_t) {{ static {cpp_type}Converter c{{}}; return &c; }});
            CPyCppyy::RegisterConverter(\"awkward::{cpp_type}&\", (cf_t)+[](CPyCppyy::cdims_t) {{ static {cpp_type}Converter c{{}}; return &c; }});
        }}

        void unregister_ak_array_view() {{
            CPyCppyy::UnregisterConverter(\"awkward::{cpp_type}\");
            CPyCppyy::UnregisterConverter(\"awkward::{cpp_type}&\");
        }}

        awkward::{cpp_type} g_ak_array_view_a, g_ak_array_view_b;
        void CallWith{cpp_type}(awkward::{cpp_type}&) {{}}
        """
        )

        # FIXME: once registered cannot unregister...
        # if hasattr(cppyy.gbl, f"{cpp_type}Converter"):

        # cppyy.gbl.unregister_ak_array_view()
        cppyy.gbl.register_ak_array_view()

    g_ak_array_view_a = cppyy.gbl.g_ak_array_view_a
    # assert g_ak_array_view_a
    # assert type(g_ak_array_view_a) == cppyy.gbl.f"{cpp_type}"
    # print("REGISTERED TYPE:", type(g_ak_array_view_a))
    #### FIXME: assert g_ak_array_view_a.wasFromMemoryCalled()

    assert not g_ak_array_view_a.wasSetArgCalled()
    # cppyy.gbl.f"CallWith{cpp_type}"(g_ak_array_view_a)
    #### FIXME: assert g_ak_array_view_a.wasSetArgCalled()

    cppyy.gbl.unregister_ak_array_view()

    g_ak_array_view_b = cppyy.gbl.g_ak_array_view_b
    # assert g_ak_array_view_b
    # assert type(g_ak_array_view_b) == cppyy.gbl.f"{cpp_type}"
    # print("REGISTERED TYPE:", type(g_ak_array_view_b))
    assert not g_ak_array_view_b.wasFromMemoryCalled()


def _array_view_type(array=None):
    pass
