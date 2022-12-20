// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_CPP_HEADERS_UTILS_H_
#define AWKWARD_CPP_HEADERS_UTILS_H_

#include <iterator>
#include <iostream>
#include <complex>
#include <type_traits>
#include <cassert>
#include <utility>
#include <stdexcept>
#include <stdint.h>

namespace awkward {

  /// @brief Returns the name of a primitive type as a string.
  template <typename T>
  const std::string
  type_to_name() {
    std::cout << "Type " << typeid(T).name() << " is not recognized." << std::endl;
    return typeid(T).name();
  }

  /// @brief Returns `bool` string when the primitive type
  /// is boolean.
  template <>
  const std::string
  type_to_name<bool>() {
    return "bool";
  }

  /// @brief Returns `int8` string when the primitive type
  /// is an 8-bit signed integer.
  template <>
  const std::string
  type_to_name<int8_t>() {
    return "int8";
  }

  /// @brief Returns `int16` string when the primitive type
  /// is a 16-bit signed integer.
  template <>
  const std::string
  type_to_name<int16_t>() {
    return "int16";
  }

  /// @brief Returns `int32` string when the primitive type
  /// is a 32-bit signed integer.
  template <>
  const std::string
  type_to_name<int32_t>() {
    return "int32";
  }

  /// @brief Returns `int64` string when the primitive type
  /// is a 64-bit signed integer.
  template <>
  const std::string
  type_to_name<int64_t>() {
    return "int64";
  }

  /// @brief Returns `int64` string when the primitive type
  /// is a 64-bit signed integer.
  template <>
  const std::string
  type_to_name<Long64_t>() {
    return "int64";
  }

  /// @brief Returns `uint8` string when the primitive type
  /// is an 8-bit unsigned integer.
  template <>
  const std::string
  type_to_name<uint8_t>() {
    return "uint8";
  }

  /// @brief Returns `uint16` string when the primitive type
  /// is a 16-bit unsigned integer.
  template <>
  const std::string
  type_to_name<uint16_t>() {
    return "uint16";
  }

  /// @brief Returns `uint32` string when the primitive type
  /// is a 32-bit unsigned integer.
  template <>
  const std::string
  type_to_name<uint32_t>() {
    return "uint32";
  }

  /// @brief Returns `uint64` string when the primitive type
  /// is a 64-bit unsigned integer.
  template <>
  const std::string
  type_to_name<uint64_t>() {
    return "uint64";
  }

  /// @brief Returns `float32` string when the primitive type
  /// is a floating point.
  template <>
  const std::string
  type_to_name<float>() {
    return "float32";
  }

  /// @brief Returns `float32` string when the primitive type
  /// is a double floating point.
  template <>
  const std::string
  type_to_name<double>() {
    return "float64";
  }

  /// @brief Returns `char` string when the primitive type
  /// is a character.
  template <>
  const std::string
  type_to_name<char>() {
    return "char";
  }

  /// @brief Returns `complex64` string when the primitive type is a
  /// complex number with float32 real and float32 imaginary parts.
  template <>
  const std::string
  type_to_name<std::complex<float>>() {
    return "complex64";
  }

  /// @brief Returns `complex128` string when the primitive type is a
  /// complex number with float64 real and float64 imaginary parts.
  template <>
  const std::string
  type_to_name<std::complex<double>>() {
    return "complex128";
  }

  /// @brief Returns `char` string when the primitive type
  /// is a character.
  template <typename T>
  const std::string
  type_to_numpy_like() {
    return type_to_name<T>();
  }

  /// @brief Returns numpy-like character code of a primitive
  /// type as a string.
  template <>
  const std::string
  type_to_numpy_like<uint8_t>() {
    return "u8";
  }

  /// @brief Returns numpy-like character code `i8`, when the
  /// primitive type is an 8-bit signed integer.
  template <>
  const std::string
  type_to_numpy_like<int8_t>() {
    return "i8";
  }

  /// @brief Returns numpy-like character code `u32`, when the
  /// primitive type is a 32-bit unsigned integer.
  template <>
  const std::string
  type_to_numpy_like<uint32_t>() {
    return "u32";
  }

  /// @brief Returns numpy-like character code `i32`, when the
  /// primitive type is a 32-bit signed integer.
  template <>
  const std::string
  type_to_numpy_like<int32_t>() {
    return "i32";
  }

  /// @brief Returns numpy-like character code `i64`, when the
  /// primitive type is a 64-bit signed integer.
  template <>
  const std::string
  type_to_numpy_like<int64_t>() {
    return "i64";
  }

  template <typename, typename = void>
  constexpr bool is_iterable{};

  // FIXME:
  // std::void_t is part of C++17, define it ourselves until we switch to it
  template <typename...>
  struct voider {
    using type = void;
  };

  template <typename... T>
  using void_t = typename voider<T...>::type;

  template <typename T>
  constexpr bool is_iterable<T,
                             void_t<decltype(std::declval<T>().begin()),
                                    decltype(std::declval<T>().end())>> = true;

  template <typename Test, template <typename...> class Ref>
  struct is_specialization : std::false_type {};

  template <template <typename...> class Ref, typename... Args>
  struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

  /// @brief Generates a Form, which is a unique description of the
  /// Layout Builder and its contents in the form of a JSON-like string.
  ///
  /// Used in RDataFrame to generate the form of the Numpy Layout Builder
  /// and ListOffset Layout Builder.
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
        parameters = std::string(
            "uint8\", \"parameters\": { \"__array__\": \"char\" }, ");
      }
      return "{\"class\": \"NumpyArray\", \"primitive\": \"" + parameters +
             "\"form_key\": \"" + form_key.str() + "\"}";
    } else if (is_specialization<T, std::complex>::value) {
      return "{\"class\": \"NumpyArray\", \"primitive\": \"" +
             type_to_name<T>() + "\", \"form_key\": \"" + form_key.str() +
             "\"}";
    }

    typedef typename T::value_type value_type;

    if (is_iterable<T>) {
      std::string parameters("");
      if (std::is_same<value_type, char>::value) {
        parameters =
            std::string(" \"parameters\": { \"__array__\": \"string\" }, ");
      }
      return "{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", "
             "\"content\":" +
             type_to_form<value_type>(form_key_id) + ", " + parameters +
             "\"form_key\": \"" + form_key.str() + "\"}";
    }
    return "unsupported type";
  }

  /// @brief Check if an RDataFrame column is an Awkward Array.
  template <typename T>
  bool
  is_awkward_type() {
    return (std::string(typeid(T).name()).find("awkward") != std::string::npos);
  }

  /// @class visit_impl
  ///
  /// @brief Class to index tuple at runtime.
  ///
  /// @tparam INDEX Index of the tuple contents.
  template <size_t INDEX>
  struct visit_impl {
    /// @brief Accesses the tuple contents at `INDEX` and
    /// calls the given function on it.
    ///
    /// @tparam CONTENT Type of tuple content.
    /// @tparam FUNCTION Function to be called on the tuple content.
    template <typename CONTENT, typename FUNCTION>
    static void
    visit(CONTENT& contents, size_t index, FUNCTION fun) {
      if (index == INDEX - 1) {
        fun(std::get<INDEX - 1>(contents));
      } else {
        visit_impl<INDEX - 1>::visit(contents, index, fun);
      }
    }
  };

  /// @brief `INDEX` reached `0`, which means the runtime index did not
  /// exist in the tuple.
  template <>
  struct visit_impl<0> {
    template <typename CONTENT, typename FUNCTION>
    static void
    visit(CONTENT& /* contents */, size_t /* index */, FUNCTION /* fun */) {
      assert(false);
    }
  };

  /// @brief Visits the tuple contents at `index`.
  template <typename FUNCTION, typename... CONTENTs>
  void
  visit_at(std::tuple<CONTENTs...> const& contents, size_t index, FUNCTION fun) {
    visit_impl<sizeof...(CONTENTs)>::visit(contents, index, fun);
  }

  /// @brief Visits the tuple contents at `index`.
  template <typename FUNCTION, typename... CONTENTs>
  void
  visit_at(std::tuple<CONTENTs...>& contents, size_t index, FUNCTION fun) {
    visit_impl<sizeof...(CONTENTs)>::visit(contents, index, fun);
  }

}  // namespace awkward

#endif  // AWKWARD_CPP_HEADERS_UTILS_H_
