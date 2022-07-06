// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UTILS_H_
#define AWKWARD_UTILS_H_

#include <iterator>
#include <complex>
#include <type_traits>
#include <cassert>
#include <utility>

namespace awkward {

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
    return "char";
}

template <>
const std::string
type_to_name<std::complex<float>>() {
    return "complex64";
}

template <>
const std::string
type_to_name<std::complex<double>>() {
    return "complex128";
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

template <typename T>
std::string
type_to_form(int64_t form_key_id) {
  if (std::string(typeid(T).name()).find("awkward") != std::string::npos) {
    return std::string("awkward type");
  }

  std::stringstream form_key;
  form_key << "node" << (form_key_id++);

  if (std::is_arithmetic<T>::value) {
    std::string parameters(type_to_name<T>() + "\", ");
    if (std::is_same<T, char>::value) {
      parameters = std::string("uint8\", \"parameters\": { \"__array__\": \"char\" }, ");
    }
    return "{\"class\": \"NumpyArray\", \"primitive\": \""
      + parameters + "\"form_key\": \"" + form_key.str() + "\"}";
  }
  else if (is_specialization<T, std::complex>::value) {
    return "{\"class\": \"NumpyArray\", \"primitive\": \""
      + type_to_name<T>() + "\", \"form_key\": \"" + form_key.str() + "\"}";
  }

  typedef typename T::value_type value_type;

  if (is_iterable<T>) {
    std::string parameters("");
    if (std::is_same<value_type, char>::value) {
      parameters = std::string(" \"parameters\": { \"__array__\": \"string\" }, ");
    }
    return "{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\":"
      + type_to_form<value_type>(form_key_id)
      + ", " + parameters + "\"form_key\": \"" + form_key.str() + "\"}";
  }
  return "unsupported type";
}

template <size_t INDEX>
struct visit_impl {
  template <typename RECORD, typename FUNCTION>
    static void visit(RECORD& contents, size_t index, FUNCTION fun) {
      if (index == INDEX - 1) {
        fun(std::get<INDEX - 1>(contents));
      }
      else {
        visit_impl<INDEX - 1>::visit(contents, index, fun);
      }
    }
};

template <>
struct visit_impl<0> {
  template <typename RECORD, typename FUNCTION>
  static void visit(RECORD& /* contents */, size_t /* index */, FUNCTION /* fun */) { assert(false); }
};

template <typename FUNCTION, typename... RECORDs>
void
visit_at(std::tuple<RECORDs...> const& contents, size_t index, FUNCTION fun) {
  visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
}

template <typename FUNCTION, typename... RECORDs>
void
visit_at(std::tuple<RECORDs...>& contents, size_t index, FUNCTION fun) {
  visit_impl<sizeof...(RECORDs)>::visit(contents, index, fun);
}

}

#endif // AWKWARD_UTILS_H_
