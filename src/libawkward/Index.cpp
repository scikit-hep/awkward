// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Index.h"

using namespace awkward;

const std::string Index::repr() const {
  std::stringstream out;
  out << "<Index [";
  if (length_ <= 10) {
    for (int i = 0;  i < length_;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
  }
  else {
    for (int i = 0;  i < 5;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
    out << " ... ";
    for (int i = length_ - 6;  i < length_;  i++) {
      if (i != length_ - 6) {
        out << " ";
      }
      out << get(i);
    }
  }
  out << "] at 0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << ">";
  return out.str();
}

IndexType Index::get(IndexType at) const {
  assert(0 <= at  &&  at < length_);
  return ptr_.get()[offset_ + at];
}

Index Index::slice(IndexType start, IndexType stop) const {
  assert(start == stop  ||  (0 <= start  &&  start < length_));
  assert(start == stop  ||  (0 < stop    &&  stop <= length_));
  assert(start <= stop);
  return Index(ptr_, offset_ + start*(start != stop), stop - start);
}
