// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <iostream>

#include "awkward/Identities.h"
#include "awkward/array/RawArray.h"

using namespace awkward;

std::string tostring(std::shared_ptr<Content> array) {
  if (RawArrayOf<double>* rawdouble = dynamic_cast<RawArrayOf<double>*>(array.get())) {
    std::stringstream out;
    out << "[";
    for (int i = 0;  i < rawdouble->length();  i++) {
      if (i != 0) out << ", ";
      out << *rawdouble->borrow(i);
    }
    out << "]";
    return out.str();
  }
  else if (RawArrayOf<int64_t>* rawint64 = dynamic_cast<RawArrayOf<int64_t>*>(array.get())) {
    std::stringstream out;
    out << "[";
    for (int i = 0;  i < rawint64->length();  i++) {
      if (i != 0) out << ", ";
      out << *rawint64->borrow(i);
    }
    out << "]";
    return out.str();
  }
  return "";
}

int main(int, char**) {
  RawArrayOf<double> rawdouble(Identities::none(), util::Parameters(), 10);
  for (int i = 0;  i < 10;  i++) {
    *rawdouble.borrow(i) = 1.1*i;
  }

  std::shared_ptr<Content> content(new RawArrayOf<double>(rawdouble));
  if (tostring(content) != "[0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]") {
    return -1;
  }
  std::shared_ptr<Content> sorted = content.get()->sort(0, true, true);
  if (tostring(sorted) != "[9.9, 8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0]") {
    return -1;
  }
  std::shared_ptr<Content> argsorted = content.get()->argsort(0, true, true);
  if (tostring(argsorted) != "[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]") {
    return -1;
  }

  return 0;
}
