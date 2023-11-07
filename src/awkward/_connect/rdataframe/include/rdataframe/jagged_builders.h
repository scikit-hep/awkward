// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
#define AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_

#include <map>
#include <string>
#include "awkward/GrowableBuffer.h"
#include "awkward/LayoutBuilder.h"
#include "awkward/utils.h"

namespace awkward {

  class CppBuffers {
  public:
    CppBuffers() = default;

    ~CppBuffers() = default;

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

    void clear() {
      map_names_nbytes_.clear();
      buffers_uint8_ptr_.clear();
    }

    template<class BUILDER, typename PRIMITIVE>
    void
    fill_from(BUILDER& builder, ROOT::RDF::RResultPtr<std::vector<PRIMITIVE>>& result) const {
      for (auto const& it : result) {
        builder.append(it);
      }
    }

    template<class BUILDER>
    size_t
    to_char_buffers(BUILDER& builder) {
      size_t length = builder.length();

      builder.to_char_buffers(buffers_uint8_ptr_);
      builder.clear();
      clear();

      return length;
    }

  private:
    std::map<std::string, size_t> map_names_nbytes_;
    std::map<std::string, uint8_t*> buffers_uint8_ptr_;
  };

}  // namespace awkward

#endif  // AWKWARD_RDATAFRAME_JAGGED_BUILDERS_H_
