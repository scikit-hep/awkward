// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
#define AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_

#include <stdlib.h>
#include <string>
#include "utils.h"


namespace awkward {

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

  int64_t
  result_distance() {
    return std::distance(result_.begin(), result_.end());
  }

  void
  fill_data_array(void* to_buffer) {
    int64_t i = 0;
    DATA* ptr = reinterpret_cast<DATA*>(to_buffer);
    for (auto const& it : result_) {
      ptr[i++] = it;
    }
  }

private:
  ROOT::RDF::RResultPtr<std::vector<T>>& result_;
  std::vector<std::vector<int64_t>> offsets_;
  std::vector<DATA> data_;
};

}

#endif // AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
