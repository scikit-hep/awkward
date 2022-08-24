// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
#define AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_

#include <stdlib.h>
#include <string>
#include "awkward/GrowableBuffer.h"
#include "awkward/LayoutBuilder.h"
#include "awkward/utils.h"

namespace awkward {

  template <typename T>
  class CppBuffers {
  public:
    CppBuffers(ROOT::RDF::RResultPtr<std::vector<T>>& result)
        : result_(result) {
    }

    ~CppBuffers() {}

    template<class BUILDER>
    const std::map<std::string, size_t>&
    names_nbytes(BUILDER& builder) {
      builder.buffer_nbytes(map_names_nbytes_);
      return map_names_nbytes_;
    }

    void
    append(const std::string& key, uint8_t* ptr) {
      buffers_uint8_ptr_[key] = ptr;
    }

    void
    check_buffers() const {
      std::cout << "CPPBuffers check buffers: " << buffers_uint8_ptr_.size() << ".";
      for (auto const& it : buffers_uint8_ptr_) {
        uint8_t* data = it.second;
        for (int i = 0; i < map_names_nbytes_[it.first]; i++) {
          std::cout << (int64_t)data[i] << ",";
        }
      }
      std::cout << std::endl;
    }

    template<class BUILDER>
    void
    fill_from(BUILDER& builder) const {
      for (auto it : result_) {
        builder.append(it);
      }
    }

    template<class BUILDER, class PRIMITIVE>
    void
    to_char_buffers(BUILDER& builder) {
      builder.to_char_buffers(buffers_uint8_ptr_);
    }

    template<class BUILDER>
    void
    fill_offsets_and_flatten_2(BUILDER& builder) const {
      for (auto const& vec : result_) {
        auto& subbuilder = builder.begin_list();
        for (auto it : vec) {
          subbuilder.append(it);
        }
        builder.end_list();
      }
    }

    template<class BUILDER>
    void
    fill_offsets_and_flatten_3(BUILDER& builder) const {
      for (auto const& vec_of_vecs : result_) {
        auto& builder1 = builder.begin_list();
        for (auto const& vec : vec_of_vecs) {
          auto& builder2 = builder1.begin_list();
          for (auto it : vec) {
            builder2.append(it);
          }
          builder1.end_list();
        }
        builder.end_list();
      }
    }

    template<class BUILDER>
    void
    fill_offsets_and_flatten_4(BUILDER& builder) const {
      for (auto const& vec_of_vecs_of_vecs : result_) {
        auto& builder1 = builder.begin_list();
        for (auto const& vec_of_vecs : vec_of_vecs_of_vecs) {
          auto& builder2 = builder1.begin_list();
          for (auto const& vec : vec_of_vecs) {
            auto& builder3 = builder2.begin_list();
            for (auto it : vec) {
              builder3.append(it);
            }
            builder2.end_list();
          }
          builder1.end_list();
        }
        builder.end_list();
      }
    }

    template<class BUILDER, typename ITERABLE>
    void
    recurse_fill_from(int64_t level, BUILDER& builder, ITERABLE& result) const {
      if (level == 0) {
        for (auto it : result) {
          builder.append(it);
        }
      }
      else {
        auto& next_builder = builder.begin_list();
        for (auto& it : result) {
          recurse_fill_from(level - 1, next_builder, it);
        }
        next_builder.end_list();
      }
    }

  private:
    ROOT::RDF::RResultPtr<std::vector<T>>& result_;
    std::map<std::string, size_t> map_names_nbytes_;
    std::map<std::string, uint8_t*> buffers_uint8_ptr_;

  };

}  // namespace awkward

#endif  // AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
