// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAY_BUILDERS_H_
#define AWKWARD_ARRAY_BUILDERS_H_

#include <iterator>
#include <stdlib.h>
#include <string>


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
    return "chars";
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

template <typename T, typename DATA>
class CppBuffers {
public:
  CppBuffers(ROOT::RDF::RResultPtr<std::vector<T>>& result)
    : result_(result) {
      offsets_.reserve(3);
      data_.reserve(1024);
    }

  ~CppBuffers() {
  }

  int64_t
  offsets_length(int64_t level) {
    return static_cast<int64_t>(offsets_[level].size());
  }

  int64_t
  data_length() {
    return data_.size();
  }

  void copy_offsets(void* to_buffer, int64_t length, int64_t level) {
    auto ptr = reinterpret_cast<int64_t *>(to_buffer);
    int64_t i = 0;
    for (auto const& it : offsets_[level]) {
      ptr[i++] = it;
    }
  }

  void copy_data(void* to_buffer, int64_t length) {
    auto ptr = reinterpret_cast<DATA*>(to_buffer);
    int64_t i = 0;
    for (auto const& it : data_) {
      ptr[i++] = it;
    }
  }

  std::pair<int64_t, int64_t>
  offsets_and_flatten_2() {
    int64_t i = 0;
    std::vector<int64_t> offsets;
    offsets.reserve(1024);
    for (auto const& vec : result_) {
      offsets.emplace_back(i);
      i += vec.size();
      data_.insert(data_.end(), vec.begin(), vec.end());
    }
    offsets.emplace_back(i);

    offsets_.emplace_back(offsets);

    return {static_cast<int64_t>(offsets_.size()), static_cast<int64_t>(offsets_[0].size())};
  }

  std::pair<int64_t, int64_t>
  offsets_and_flatten_3() {
    int64_t i = 0;
    int64_t j = 0;
    std::vector<int64_t> offsets;
    offsets.reserve(1024);
    std::vector<int64_t> inner_offsets;
    inner_offsets.reserve(1024);
    for (auto const& vec_of_vecs : result_) {
      offsets.emplace_back(i);
      i += vec_of_vecs.size();

      for (auto const& vec : vec_of_vecs) {
        inner_offsets.emplace_back(j);
        j += vec.size();
        data_.insert(data_.end(), vec.begin(), vec.end());
      }
      inner_offsets.emplace_back(j);
    }
    offsets.emplace_back(i);

    offsets_.emplace_back(offsets);
    offsets_.emplace_back(inner_offsets);

    return {static_cast<int64_t>(offsets_.size()), static_cast<int64_t>(offsets_[0].size())};
  }

  std::pair<int64_t, int64_t>
  offsets_and_flatten_4() {
    int64_t i = 0;
    int64_t j = 0;
    int64_t k = 0;
    std::vector<int64_t> offsets;
    std::vector<int64_t> inner_offsets;
    std::vector<int64_t> inner_inner_offsets;
    for (auto const& vec_of_vecs_of_vecs : result_) {
      offsets.emplace_back(i);
      i += vec_of_vecs_of_vecs.size();

      for (auto const& vec_of_vecs : vec_of_vecs_of_vecs) {
        inner_offsets.emplace_back(j);
        j += vec_of_vecs.size();

        for (auto const&vec : vec_of_vecs) {
          inner_inner_offsets.emplace_back(k);
          k += vec.size();
          data_.insert(data_.end(), vec.begin(), vec.end());
        }
        inner_inner_offsets.emplace_back(k);
      }
      inner_offsets.emplace_back(j);
    }
    offsets.emplace_back(i);

    offsets_.emplace_back(offsets);
    offsets_.emplace_back(inner_offsets);
    offsets_.emplace_back(inner_inner_offsets);

    return {static_cast<int64_t>(offsets_.size()), static_cast<int64_t>(offsets_[0].size())};
  }

  std::pair<int64_t, void*>
  create_array() {
    int64_t size = result_->size();
    DATA* ptr = new DATA[size];
    int64_t i = 0;
    for (auto const& it : result_) {
      ptr[i++] = it;
    }
    return {size, ptr};
  }

private:
  ROOT::RDF::RResultPtr<std::vector<T>>& result_;
  std::vector<std::vector<int64_t>> offsets_;
  std::vector<DATA> data_;
};

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
