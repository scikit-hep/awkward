# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

import ROOT
import cppyy

cppyy.add_include_path("include")

compiler = ROOT.gInterpreter.Declare


compiler(
    """
#include <iterator>
#include <stdlib.h>
#include <string>

extern "C" {
    uint8_t
    awkward_ArrayBuilder_boolean(void* arraybuilder,
                                 bool x);

    uint8_t
    awkward_ArrayBuilder_integer(void* arraybuilder,
                                 int64_t x);

    uint8_t
    awkward_ArrayBuilder_real(void* arraybuilder,
                              double x);

    uint8_t
    awkward_ArrayBuilder_complex(void* arraybuilder,
                                 double real,
                                 double imag);

    uint8_t
    awkward_ArrayBuilder_string(void* arraybuilder,
                                const char* x);

    uint8_t
    awkward_ArrayBuilder_beginlist(void* arraybuilder);

    uint8_t
    awkward_ArrayBuilder_endlist(void* arraybuilder);
}

namespace awkward {

template <typename T>
struct build_array_impl {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        cout << "FIXME: processing of a " << typeid(T).name()
            << " type is not implemented yet." << endl;
    };
};

template <>
struct build_array_impl<bool> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<bool>>& result, void* ptr) {
        for (auto const& it : result) {
            awkward_ArrayBuilder_boolean(ptr, it);
        }
    };
};

template <>
struct build_array_impl<int64_t> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<int64_t>>& result, void* ptr) {
        for (auto const& it : result) {
            awkward_ArrayBuilder_integer(ptr, it);
        }
    };
};

template <>
struct build_array_impl<double> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<double>>& result, void* ptr) {
        for (auto const& it : result) {
            awkward_ArrayBuilder_real(ptr, it);
        }
    };
};

template <>
struct build_array_impl<std::complex<double>> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<std::complex<double>>>& result, void* ptr) {
        for (auto const& it : result) {
            awkward_ArrayBuilder_complex(ptr, it.real(), it.imag());
        }
    };
};

template <>
struct build_array_impl<std::string> {
    static void build_array(ROOT::RDF::RResultPtr<std::vector<std::string>>& result, void* ptr) {
        for (auto const& it : result) {
            awkward_ArrayBuilder_string(ptr, it.c_str());
        }
    };
};

template <typename T>
void
build_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<void *>(builder_ptr);
    build_array_impl<T>::build_array(result, ptr);
}

template <typename T, typename V>
struct build_list_array_impl {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
        typedef typename T::value_type value_type;

        cout << "FIXME: processing an iterable of a " << typeid(value_type).name()
            << " type is not implemented yet." << endl;
    };
};

template <typename T>
struct build_list_array_impl<T, double> {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        for (auto const& data : result) {
            awkward_ArrayBuilder_beginlist(ptr);
            for (auto const& it : data) {
                awkward_ArrayBuilder_real(ptr, it);
            }
            awkward_ArrayBuilder_endlist(ptr);
        }
    };
};


template <typename T>
struct build_list_array_impl<T, ROOT::VecOps::RVec<ROOT::VecOps::RVec<double> > > {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        typedef typename T::value_type value_type;

        cout << "FIXME: processing an iterable of a " << typeid(value_type).name()
            << " type is not implemented yet." << endl;
    };
};

template <typename T>
struct build_list_array_impl<T, int64_t> {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
    for (auto const& data : result) {
        awkward_ArrayBuilder_beginlist(ptr);
        for (auto const& it : data) {
            awkward_ArrayBuilder_integer(ptr, it);
        }
        awkward_ArrayBuilder_endlist(ptr);
    }
    };
};

template <typename T>
struct build_list_array_impl<T, bool> {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        for (auto const& data : result) {
            awkward_ArrayBuilder_beginlist(ptr);
            for (auto const& it : data) {
                awkward_ArrayBuilder_boolean(ptr, it);
            }
            awkward_ArrayBuilder_endlist(ptr);
        }
    };
};

template <typename T>
struct build_list_array_impl<T, std::complex<double>> {
    static void build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
        for (auto const& data : result) {
            awkward_ArrayBuilder_beginlist(ptr);
            for (auto const& it : data) {
                awkward_ArrayBuilder_complex(ptr, it.real(), it.imag());
            }
            awkward_ArrayBuilder_endlist(ptr);
        }
    };
};

template <typename T>
void
build_list_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<void *>(builder_ptr);
    build_list_array_impl<T, typename T::value_type>::build_list_array(result, ptr);
}

template <>
void
build_list_array<std::string>(ROOT::RDF::RResultPtr<std::vector<std::string>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<void *>(builder_ptr);
    for (auto const& it : result) {
        awkward_ArrayBuilder_string(ptr, it.c_str());
    }
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

template <typename T>
void
offsets_and_flatten_impl(
    T& result,
    void* ptr)
{
    typedef typename T::value_type value_type;

    cout << "FIXME: processing an iterable of a " << typeid(value_type).name()
        << " type is not implemented yet." << endl;
}

template<>
void
offsets_and_flatten_impl<ROOT::VecOps::RVec<ROOT::VecOps::RVec<double> > const> (
    const ROOT::VecOps::RVec<ROOT::VecOps::RVec<double> >& result,
    void* ptr)
{
    std::for_each(result.begin(), result.end(), [&] (auto const& n) {
        awkward_ArrayBuilder_beginlist(ptr);
        for (auto const& it : n) {
            awkward_ArrayBuilder_real(ptr, it);
        }
        awkward_ArrayBuilder_endlist(ptr);
    });
}

template <typename T>
void
offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<void *>(builder_ptr);

    typedef typename T::value_type value_type;

    if (is_iterable<value_type>) {
        std::for_each(result->begin(), result->end(), [&] (auto const& n) {
            awkward_ArrayBuilder_beginlist(ptr);
            offsets_and_flatten_impl(n, ptr);
            awkward_ArrayBuilder_endlist(ptr);
        });
    }
}

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {
};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {
};

template <typename T, typename std::enable_if<is_specialization<T, std::complex>::value, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    build_array(result, builder_ptr);
    return std::string("complex");
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    build_array(result, builder_ptr);
    return std::string("primitive");
}

template <typename T, typename std::enable_if<is_iterable<T>, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {

    auto str = std::string(typeid(T).name());

    if (str.find("awkward") != string::npos) {
        return std::string("awkward type");
    }
    else {

        typedef typename T::value_type value_type;

        if (is_iterable<value_type>) {
            return std::string("nested iterable");
        } else if (std::is_arithmetic<value_type>::value) {
            build_list_array(result, builder_ptr);
            return std::string("iterable");
        } else if (is_specialization<value_type, std::complex>::value) {
            build_list_array(result, builder_ptr);
            return "iterable of complex";
        }
        return "something_else";
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

    def _maybe_wrap(array, column_as_record):
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
        )

    # Cast input node to base RNode type
    data_frame_rnode = cppyy.gbl.ROOT.RDF.AsRNode(data_frame)

    column_type = data_frame_rnode.GetColumnType(column)

    # 'Take' is a lazy action:
    result_ptrs = data_frame_rnode.Take[column_type](column)
    builder = ak._v2.highlevel.ArrayBuilder()

    ptrs_type = ROOT.awkward.check_type_of[column_type](
        result_ptrs, builder._layout._ptr
    )

    if ptrs_type in ("primitive", "complex", "iterable", "iterable of complex"):

        return _maybe_wrap(builder.snapshot(), column_as_record)

    elif ptrs_type == "nested iterable":

        ROOT.awkward.offsets_and_flatten[column_type](result_ptrs, builder._layout._ptr)
        
        return _maybe_wrap(builder.snapshot(), column_as_record)

    elif ptrs_type == "awkward type":

        # Triggers event loop and execution of all actions booked in the associated RLoopManager.
        cpp_reference = result_ptrs.GetValue()

        return _wrap_as_array(column, cpp_reference, column_as_record)
    else:
        raise ak._v2._util.error(NotImplementedError)
