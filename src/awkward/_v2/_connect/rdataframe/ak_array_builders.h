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

template<typename T>
std::vector<T>
vect_array(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  std::vector<T> vect;
  vect.assign(result.begin(), result.end());
  return vect;
}

template<typename T>
void
fill_array(ROOT::RDF::RResultPtr<std::vector<T>>& result, unsigned char* array) {
  auto ptr = reinterpret_cast<T*>(array);
  int64_t i = 0;
  for (auto const& it : result) {
    ptr[i++] = it;
  }
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

template <typename, typename = void>
constexpr bool is_iterable{};

template <typename T>
constexpr bool is_iterable<
    T,
    std::void_t< decltype(std::declval<T>().begin()),
                 decltype(std::declval<T>().end())
    >
> = true;


template <typename T, typename V>
struct copy_offsets_and_flatten_impl {
    static void
      copy_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result, std::vector <std::pair<void*, size_t>>& dict) {

        typedef typename T::value_type value_type;

        size_t offsets_length = result->size() + 1;
        int64_t* offsets = (int64_t*)malloc(sizeof(int64_t)*offsets_length);
        int64_t i = 1;
        offsets[0] = 0;
        for (auto const& it : result) {
          offsets[i] = offsets[i - 1] + it.size();
          i++;
        }
        dict.push_back({offsets, offsets_length - 1});

        size_t data_length = offsets[i - 1];
        V* data = (V*)malloc(sizeof(V)*data_length);
        int64_t j = 0;
        for (auto const& vec : result) {
          for (auto const& x : vec) {
            data[j++] = x;
          }
        }
        dict.push_back({data, data_length});
    };
};

template <typename T>
std::vector<std::pair<void*, size_t>>
copy_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
  std::vector <std::pair<void*, size_t>> dict;
  copy_offsets_and_flatten_impl<T, typename T::value_type>::copy_offsets_and_flatten(result, dict);
  return dict;
}

// template <typename T, typename V>
// struct copy_nested_offsets_and_flatten_impl {
//     static void
//       copy_nested_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result, std::vector <std::vector<V>>& dict) {
//
//         std::vector<int64_t> offsets;
//         std::vector<int64_t> inner_offsets;
//         std::vector<V> data;
//
//         for (auto const& vec_of_vecs : result) {
//           offsets.emplace_back(vec_of_vecs.size());
//           for (auto const& vec : vec_of_vecs) {
//             inner_offsets.emplace_back(vec.size());
//             for (auto const& x : vec) {
//               data.emplace_back(x);
//             }
//           }
//         }
//         dict.push_back(offsets);
//         dict.push_back(inner_offsets);
//         dict.push_back(data);
//     };
// };
//
// template <typename T, typename V>
// std::vector<std::vector<V>>
// copy_nested_offsets_and_flatten(ROOT::RDF::RResultPtr<std::vector<T>>& result) {
//   typedef typename T::value_type value_type;
//   typedef typename value_type::value_type value_value_type;
//   std::cout << typeid(T).name() << std::endl;
//   std::cout << typeid(value_type).name() << std::endl;
//   std::cout << typeid(value_value_type).name() << std::endl;
//   std::vector<std::vector<V>> dict;
//
//   copy_nested_offsets_and_flatten_impl<T, value_value_type>::copy_nested_offsets_and_flatten(result, dict);
//   return dict;
// }

template <typename T>
class Buffers {
public:
  Buffers(ROOT::RDF::RResultPtr<std::vector<T>>& result)
    : result_(result) {}

  ~Buffers() {
    std::cout << "Buffers destructed!\n" << std::endl;
  }

  std::vector<std::pair<int64_t, void*>>
  offsets_and_flatten() {

    int64_t i = 0;
    int64_t j = 0;
    for (auto const& vec_of_vecs : result_) {
      offsets_.emplace_back(i);
      i += vec_of_vecs.size();

      for (auto const& vec : vec_of_vecs) {
        inner_offsets_.emplace_back(j);
        j += vec.size();

        for (auto const& x : vec) {
          data_.emplace_back(x);
        }
      }
      inner_offsets_.emplace_back(j);
    }
    offsets_.emplace_back(i);

    return {
      {offsets_.size(), reinterpret_cast<void *>(&offsets_[0])},
      {inner_offsets_.size(), reinterpret_cast<void *>(&inner_offsets_[0])},
      {data_.size(), reinterpret_cast<void *>(&data_[0])}
    };
  }

private:
  ROOT::RDF::RResultPtr<std::vector<T>>& result_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> inner_offsets_;
  std::vector<double> data_;
};

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

template <typename T>
std::string
type_to_form(int64_t form_key_id) {
  if (std::string(typeid(T).name()).find("awkward") != string::npos) {
    return std::string("awkward type");
  }

  std::stringstream form_key;
  form_key << "node" << (form_key_id++);

  if (std::is_arithmetic<T>::value) {
    std::string parameters(type_to_name<T>() + "\",");
    if (std::is_same<T, char>::value) {
      parameters = std::string("uint8\", \"parameters\": { \"__array__\": \"char\" }, ");
    }
    return "{\"class\": \"NumpyArray\", \"primitive\": \""
      + parameters + "\"form_key\": \"" + form_key.str() + "\"}";
  } else if (is_specialization<T, std::complex>::value) {
    return "{\"class\": \"NumpyArray\", \"primitive\": \"complex128\", \"form_key\": \""
      + form_key.str() + "\"}";
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

}

#endif // AWKWARD_ARRAY_BUILDERS_H_
