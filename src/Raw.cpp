// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Raw.h"

using awkward;

std::string Index::repr() {
  std::stringstream out;
  out << "<Raw '" << std::hex;
  if (len() <= 40) {
    for (int i = 0;  i < len();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
  }
  else {
    for (int i = 0;  i < 20;  i++) {
      if (i != 0) {
        out << " ";
      }
      out << get(i);
    }
    out << " ... ";
    for (int i = len() - 21;  i < len();  i++) {
      if (i != len() - 21) {
        out << " ";
      }
      out << get(i);
    }
  }
  out << "' at 0x";
  out << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << ">";
  return out.str();
}

RawType Raw::getraw(IndexType at) {
  assert(0 <= at  &&  at < entrysize_*length_);
  return ptr_.get()[entrysize_*offset_ + at];
}

Raw Raw::slice(IndexType start, IndexType stop) {
  assert(start == stop  ||  (0 <= start  &&  start < length_));
  assert(start == stop  ||  (0 < stop    &&  stop <= length_));
  assert(start <= stop);
  return Raw(ptr_, offset_ + start*(start != stop), stop - start);
}
