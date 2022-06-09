// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAY_BUILDERS_H_
#define AWKWARD_ARRAY_BUILDERS_H_

#include "awkward/layoutbuilder/LayoutBuilder.h"

#include <iterator>
#include <stdlib.h>
#include <string>

extern "C" {
    uint8_t
    awkward_ArrayBuilder_boolean(void* array_builder,
                                 bool x);

    uint8_t
    awkward_ArrayBuilder_integer(void* array_builder,
                                 int64_t x);

    uint8_t
    awkward_ArrayBuilder_real(void* array_builder,
                              double x);

    uint8_t
    awkward_ArrayBuilder_complex(void* array_builder,
                                 double real,
                                 double imag);

    uint8_t
    awkward_ArrayBuilder_string(void* array_builder,
                                const char* x);

    uint8_t
    awkward_ArrayBuilder_beginlist(void* array_builder);

    uint8_t
    awkward_ArrayBuilder_endlist(void* array_builder);
}

namespace awkward {

template<typename T>
void*
create_array(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  T* ptr = (T*)malloc(sizeof(T)*result->size());
  int64_t i = 0;
  for (auto const& it : result) {
    ptr[i++] = it;
  }
  return ptr;
}

template <typename T>
const std::string
type_to_name() {
    return typeid(T).name();
}

template <>
const std::string
type_to_name<bool>() {
    return "bool";
}

template <>
const std::string
type_to_name<int8_t>() {
    return "int8";
}

template <>
const std::string
type_to_name<int16_t>() {
    return "int16";
}

template <>
const std::string
type_to_name<int32_t>() {
    return "int32";
}

template <>
const std::string
type_to_name<int64_t>() {
    return "int64";
}

template <>
const std::string
type_to_name<uint8_t>() {
    return "uint8";
}

template <>
const std::string
type_to_name<uint16_t>() {
    return "uint16";
}

template <>
const std::string
type_to_name<uint32_t>() {
    return "uint32";
}

template <>
const std::string
type_to_name<uint64_t>() {
    return "uint64";
}

template <>
const std::string
type_to_name<float>() {
    return "float32";
}

template <>
const std::string
type_to_name<double>() {
    return "float64";
}

template <>
const std::string
type_to_name<char>() {
    return "chars";
}

template <typename T>
struct build_layout_impl {
    static void build_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, awkward::LayoutBuilder64* builder) {
        cout << "FIXME: processing of a " << typeid(T).name()
            << " type is not implemented yet." << endl;
    };
};

template <>
struct build_layout_impl<int64_t> {
    static void build_layout(ROOT::RDF::RResultPtr<std::vector<int64_t>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& it : result) {
            builder->int64(it);
        }
    };
};

template <>
struct build_layout_impl<double> {
    static void build_layout(ROOT::RDF::RResultPtr<std::vector<double>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& it : result) {
            builder->float64(it);
        }
    };
};

template <>
struct build_layout_impl<std::complex<double>> {
    static void build_layout(ROOT::RDF::RResultPtr<std::vector<std::complex<double>>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& it : result) {
            builder->complex(it);
        }
    };
};

template <typename T>
void
build_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<awkward::LayoutBuilder64 *>(builder_ptr);
    build_layout_impl<T>::build_layout(result, ptr);
}

template <typename T, typename V>
struct build_list_offset_layout_impl {
    static void build_list_offset_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
        typedef typename T::value_type value_type;

        cout << "FIXME: processing an iterable of a " << typeid(value_type).name()
            << " type is not implemented yet." << endl;
    };
};

template <typename T>
struct build_list_offset_layout_impl<T, int64_t> {
    static void build_list_offset_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& data : result) {
            builder->begin_list();
            for (auto const& x : data) {
                builder->int64(x);
            }
            builder->end_list();
        }
    };
};

template <typename T>
struct build_list_offset_layout_impl<T, double> {
    static void build_list_offset_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& data : result) {
            builder->begin_list();
            for (auto const& x : data) {
                builder->float64(x);
            }
            builder->end_list();
        }
    };
};

template <typename T>
struct build_list_offset_layout_impl<T, std::complex<double>> {
    static void build_list_offset_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, awkward::LayoutBuilder64* builder) {
        for (auto const& data : result) {
            builder->begin_list();
            for (auto const& x : data) {
                builder->complex(x);
            }
            builder->end_list();
        }
    };
};

template <typename T>
void
build_list_offset_layout(ROOT::RDF::RResultPtr<std::vector<T>>& result, long builder_ptr) {
    auto ptr = reinterpret_cast<awkward::LayoutBuilder64 *>(builder_ptr);
    build_list_offset_layout_impl<T, typename T::value_type>::build_list_offset_layout(result, ptr);
}

template <>
void
build_list_offset_layout<std::string>(ROOT::RDF::RResultPtr<std::vector<std::string>>& result, long builder_ptr) {
    auto builder = reinterpret_cast<awkward::LayoutBuilder64 *>(builder_ptr);
    for (auto const& it : result) {
        builder->begin_list();
        builder->string(it.c_str(), it.length());
        builder->end_list();
    }
}

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

// template <typename T>
// struct copy_array_impl {
//     static void* copy_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, void* ptr) {
//         cout << "FIXME: processing of a " << typeid(T).name()
//             << " type is not implemented yet." << endl;
//     };
// };
//
// template <>
// struct copy_array_impl<bool> {
//     static void* copy_array(ROOT::RDF::RResultPtr<std::vector<bool>>& result, void* ptr) {
//
//         for (auto const& it : result) {
//             awkward_ArrayBuilder_boolean(ptr, it);
//         }
//     };
// };
//
// template <>
// struct copy_array_impl<int64_t> {
//     static void copy_array(ROOT::RDF::RResultPtr<std::vector<int64_t>>& result, void* ptr) {
//         for (auto const& it : result) {
//             awkward_ArrayBuilder_integer(ptr, it);
//         }
//     };
// };
//
// template <>
// struct copy_array_impl<double> {
//     static void copy_array(ROOT::RDF::RResultPtr<std::vector<double>>& result, void* ptr) {
//         for (auto const& it : result) {
//             awkward_ArrayBuilder_real(ptr, it);
//         }
//     };
// };
//
// template <>
// struct copy_array_impl<std::complex<double>> {
//     static void copy_array(ROOT::RDF::RResultPtr<std::vector<std::complex<double>>>& result, void* ptr) {
//         for (auto const& it : result) {
//             awkward_ArrayBuilder_complex(ptr, it.real(), it.imag());
//         }
//     };
// };
//
// template <>
// struct copy_array_impl<std::string> {
//     static void* copy_array(ROOT::RDF::RResultPtr<std::vector<std::string>>& result) {
//       return create_array(result, result.size());
//         // for (auto const& it : result) {
//         //     awkward_ArrayBuilder_string(ptr, it.c_str());
//         // }
//     };
// };

template <typename T>
std::pair<void*, size_t>
copy_array(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  return {create_array<T>(result), result->size()};
  //  copy_array_impl<T>::copy_array(result);
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
    T & result,
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

template<>
void
offsets_and_flatten_impl<ROOT::VecOps::RVec<ROOT::VecOps::RVec<int64_t> > const> (
    const ROOT::VecOps::RVec<ROOT::VecOps::RVec<int64_t> >& result,
    void* ptr)
{
    std::for_each(result.begin(), result.end(), [&] (auto const& n) {
        awkward_ArrayBuilder_beginlist(ptr);
        for (auto const& it : n) {
            awkward_ArrayBuilder_integer(ptr, it);
        }
        awkward_ArrayBuilder_endlist(ptr);
    });
}

template<>
void
offsets_and_flatten_impl<ROOT::VecOps::RVec<ROOT::VecOps::RVec<std::complex<double>> > const> (
    const ROOT::VecOps::RVec<ROOT::VecOps::RVec<std::complex<double>> >& result,
    void* ptr)
{
    std::for_each(result.begin(), result.end(), [&] (auto const& n) {
        awkward_ArrayBuilder_beginlist(ptr);
        for (auto const& it : n) {
            awkward_ArrayBuilder_complex(ptr, it.real(), it.imag());
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
        typedef typename value_type::value_type value_value_type;

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
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
    return std::string("complex128");
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
    return type_to_name<T>();
}

template <typename T, typename std::enable_if<is_iterable<T>, T>::type * = nullptr>
std::string
check_type_of(ROOT::RDF::RResultPtr<std::vector<T>>& result) {

    auto str = std::string(typeid(T).name());

    if (str.find("awkward") != string::npos) {
        return std::string("awkward type");
    }
    else {

        typedef typename T::value_type value_type;

        if (is_iterable<value_type>) {
            return std::string("nested iterable");
        } else if (std::is_arithmetic<value_type>::value) {
            return std::string("iterable " + type_to_name<value_type>());
        } else if (is_specialization<value_type, std::complex>::value) {
            return std::string("iterable ") + std::string("complex128");
        }
        return "something_else";
    }
    return "undefined";
}
}

#endif // AWKWARD_ARRAY_BUILDERS_H_
