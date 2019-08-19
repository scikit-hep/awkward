// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <memory>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  typedef int32_t INDEXTYPE;

  class Index {
  public:
    Index(ssize_t length);
    Index(std::shared_ptr<INDEXTYPE> data, ssize_t length);
    Index(std::shared_ptr<INDEXTYPE> data, ssize_t length, ssize_t stride);

    INDEXTYPE GetItem(ssize_t slot);

  private:
    std::shared_ptr<INDEXTYPE> data_;
    ssize_t length_;
    ssize_t stride_;
  };
}

#endif // AWKWARD_INDEX_H_
