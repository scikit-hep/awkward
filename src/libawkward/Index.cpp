// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Index.h"

namespace ak = awkward;

ak::Index::Index(ssize_t length) {
  data_ = std::shared_ptr<ak::INDEXTYPE>(new ak::INDEXTYPE[length], std::default_delete<ak::INDEXTYPE[]>());
  for (ssize_t i = 0;  i < length;  i++) data_.get()[i] = i;  // FIXME
  length_ = length;
  stride_ = sizeof(INDEXTYPE);
}

ak::Index::Index(std::shared_ptr<ak::INDEXTYPE> data, ssize_t length) {
  data_ = data;
  length_ = length;
  stride_ = sizeof(INDEXTYPE);
}

ak::Index::Index(std::shared_ptr<ak::INDEXTYPE> data, ssize_t length, ssize_t stride) {
  data_ = data;
  length_ = length;
  stride_ = stride;
}

ak::INDEXTYPE ak::Index::GetItem(ssize_t slot) {
  return *reinterpret_cast<ak::INDEXTYPE*>(reinterpret_cast<ssize_t>(data_.get()) + slot*stride_);
}
