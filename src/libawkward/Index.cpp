// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Index.h"

using namespace awkward;

const std::string Index::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<Index i=\"[";
  if (length_ <= 10) {
    for (IndexType i = 0;  i < length_;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
  }
  else {
    for (IndexType i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
    out << " ... ";
    for (IndexType i = length_ - 6;  i < length_;  i++) {
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

IndexType Index::get(AtType at) const {
  assert(0 <= at  &&  at < length_);
  return ptr_.get()[offset_ + at];
}

Index Index::slice(AtType start, AtType stop) const {
  assert(start == stop  ||  (0 <= start  &&  start < length_));
  assert(start == stop  ||  (0 < stop    &&  stop <= length_));
  assert(start <= stop);
  // return Index(ptr_, offset_ + start*(start != stop), stop - start);
  throw new std::invalid_argument("");
}
