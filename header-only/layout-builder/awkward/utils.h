// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

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
#include <typeinfo>
#include <map>
#include <vector>

namespace awkward {

  // FIXME:
  // The following helper variable templates are part of C++17,
  // define it ourselves until we switch to it
  template< class T >
  constexpr bool is_integral_v = std::is_integral<T>::value;

  template< class T >
  constexpr bool is_signed_v = std::is_signed<T>::value;

  template< class T, class U >
  constexpr bool is_same_v = std::is_same<T, U>::value;

  /// @brief Returns the name of a primitive type as a string.
  template <typename T>
  inline const std::string
  type_to_name() {
    if (is_integral_v<T>) {
      if (is_signed_v<T>) {
        if (sizeof(T) == 1) {
          return "int8";
        }
        else if (sizeof(T) == 2) {
          return "int16";
        }
        else if (sizeof(T) == 4) {
          return "int32";
        }
        else if (sizeof(T) == 8) {
          return "int64";
        }
      }
      else {
        if (sizeof(T) == 1) {
          return "uint8";
        }
        else if (sizeof(T) == 2) {
          return "uint16";
        }
        else if (sizeof(T) == 4) {
          return "uint32";
        }
        else if (sizeof(T) == 8) {
          return "uint64";
        }
      }
    }
    else if (is_same_v<T, float>) {
      return "float32";
    }
    else if (is_same_v<T, double>) {
      return "float64";
    }
    else if (is_same_v<T, std::complex<float>>) {
      return "complex64";
    }
    else if (is_same_v<T, std::complex<double>>) {
      return "complex128";
    }

    // std::is_integral_v<T> and sizeof(T) not in (1, 2, 4, 8) can get here.
    // Don't connect this line with the above as an 'else' clause.
    return std::string("unsupported primitive type: ") + typeid(T).name();
  }

  template <>
  inline const std::string
  type_to_name<bool>() {
    // This takes precedence over the unspecialized template, and therefore any
    // 8-bit data that is not named bool will be mapped to "int8" or "uint8".
    return "bool";
  }

  template <>
  inline const std::string
  type_to_name<char>() {
    // This takes precedence over the unspecialized template, and therefore any
    // 8-bit data that is not named char will be mapped to "int8" or "uint8".
    return "char";
  }


  /// @brief Returns `char` string when the primitive type
  /// is a character.
  template <typename T>
  inline const std::string
  type_to_numpy_like() {
    return type_to_name<T>();
  }

  /// @brief Returns numpy-like character code of a primitive
  /// type as a string.
  template <>
  inline const std::string
  type_to_numpy_like<uint8_t>() {
    return "u8";
  }

  /// @brief Returns numpy-like character code `i8`, when the
  /// primitive type is an 8-bit signed integer.
  template <>
  inline const std::string
  type_to_numpy_like<int8_t>() {
    return "i8";
  }

  /// @brief Returns numpy-like character code `u32`, when the
  /// primitive type is a 32-bit unsigned integer.
  template <>
  inline const std::string
  type_to_numpy_like<uint32_t>() {
    return "u32";
  }

  /// @brief Returns numpy-like character code `i32`, when the
  /// primitive type is a 32-bit signed integer.
  template <>
  inline const std::string
  type_to_numpy_like<int32_t>() {
    return "i32";
  }

  /// @brief Returns numpy-like character code `i64`, when the
  /// primitive type is a 64-bit signed integer.
  template <>
  inline const std::string
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
  template <typename T, typename OFFSETS>
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
      return "{\"class\": \"ListOffsetArray\", \"offsets\": \"" +
             type_to_numpy_like<OFFSETS>() + "\", "
             "\"content\":" +
             type_to_form<value_type, OFFSETS>(form_key_id) + ", " + parameters +
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

  /// @brief Helper function to retrieve the names of the buffers.
  ///
  /// Note: use with caution, beware of a potential mismatch between retrieved values!
  template<typename LayoutBuilder>
  std::vector<std::string> buffer_name_helper(const LayoutBuilder* builder) {
    std::map <std::string, size_t> names_nbytes = {};
    std::vector<std::string> buffer_name;
    builder->buffer_nbytes(names_nbytes);
    for (auto it: names_nbytes) {
      buffer_name.push_back(it.first);
    }
    return buffer_name;
  }

  /// @brief Helper function to retrieve the sizes (in bytes) of the buffers.
  ///
  /// Note: use with caution, beware of a potential mismatch between retrieved values!
  template<typename LayoutBuilder>
  std::vector<size_t> buffer_size_helper(const LayoutBuilder* builder) {
    std::map <std::string, size_t> names_nbytes = {};
    std::vector<size_t> buffer_size;
    builder->buffer_nbytes(names_nbytes);
    for (auto it: names_nbytes) {
      buffer_size.push_back(it.second);
    }
    return buffer_size;
  }

  /// @brief Helper function to retrieve the number of the buffers.
  ///
  /// Note: use with caution, beware of a potential mismatch between retrieved values!
  template<typename LayoutBuilder>
  size_t num_buffers_helper(const LayoutBuilder* builder) {
    std::map <std::string, size_t> names_nbytes = {};
    builder->buffer_nbytes(names_nbytes);
    return names_nbytes.size();
  }

}  // namespace awkward

#endif  // AWKWARD_CPP_HEADERS_UTILS_H_
