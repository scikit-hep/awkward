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

    template<class BUILDER, typename PRIMITIVE>
    void
    fill_from(BUILDER& builder, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) const {
      for (auto it : result) {
        builder.append(it);
      }
    }

    template<class BUILDER>
    void
    to_char_buffers(BUILDER& builder) {
      builder.to_char_buffers(buffers_uint8_ptr_);
    }

  private:
    ROOT::RDF::RResultPtr<std::vector<T>>& result_;
    std::map<std::string, size_t> map_names_nbytes_;
    std::map<std::string, uint8_t*> buffers_uint8_ptr_;

  };

}  // namespace awkward

#endif  // AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
