// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAY_BUILDERS_H_
#define AWKWARD_ARRAY_BUILDERS_H_

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
std::pair<void*, size_t>
copy_buffer(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  return {create_array<T>(result), result->size()};
}

template <typename T, typename V>
struct copy_offsets_and_flatten_impl {
    static std::vector<std::pair<void*, size_t>>
      copy_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
        //typedef typename T::value_type value_type;

        size_t offsets_length = result->size() + 1;
        int64_t* offsets = (int64_t*)malloc(sizeof(int64_t)*offsets_length);
        int64_t i = 1;
        offsets[0] = 0;
        for (auto const& it : result) {
          offsets[i] = offsets[i - 1] + it.size();
          i++;
        }

        size_t data_length = offsets[i - 1];
        V* data = (V*)malloc(sizeof(V)*data_length);
        int64_t j = 0;
        for (auto const& vec : result) {
          for (auto const& x : vec) {
            data[j++] = x;
          }
        }
        return {{offsets, offsets_length - 1}, {data, data_length}};
    };
};

template <typename T>
std::vector<std::pair<void*, size_t>>
copy_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  return copy_offsets_and_flatten_impl<T, typename T::value_type>::copy_offsets_and_flatten(result);
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
