// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Index.h"

using namespace awkward;

template <typename T>
const std::string IndexOf<T>::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<Index i=\"[";
  if (length_ <= 10) {
    for (T i = 0;  i < length_;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
  }
  else {
    for (T i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
    out << " ... ";
    for (T i = length_ - 6;  i < length_;  i++) {
      if (i != length_ - 6) {
        out << " ";
      }
      out << get(i);
    }
  }
  out << "]\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

template <typename T>
T IndexOf<T>::get(T at) const {
  assert(0 <= at  &&  at < length_);
  return ptr_.get()[offset_ + at];
}

template <typename T>
IndexOf<T> IndexOf<T>::slice(T start, T stop) const {
  assert(start == stop  ||  (0 <= start  &&  start < length_));
  assert(start == stop  ||  (0 < stop    &&  stop <= length_));
  assert(start <= stop);
  return IndexOf<T>(ptr_, offset_ + start*(start != stop), stop - start);
}

namespace awkward {
  template class IndexOf<IndexType>;
  template class IndexOf<TagType>;
  template class IndexOf<ChunkOffsetType>;
}
