// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/Identity.h"
#include "awkward/RawArray.h"
#include "awkward/ListArray.h"
#include "awkward/ListOffsetArray.h"
#include "awkward/Slice.h"

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
  else if (ListOffsetArray32* listoffset = dynamic_cast<ListOffsetArray32*>(array.get())) {
    std::stringstream out;
    out << "[";
    for (int i = 0;  i < listoffset->length();  i++) {
      if (i != 0) out << ", ";
      out << tostring(listoffset->getitem_at_unsafe(i));
    }
    out << "]";
    return out.str();
  }
  else if (ListArray32* list = dynamic_cast<ListArray32*>(array.get())) {
    std::stringstream out;
    out << "[";
    for (int i = 0;  i < list->length();  i++) {
      if (i != 0) out << ", ";
      out << tostring(list->getitem_at_unsafe(i));
    }
    out << "]";
    return out.str();
  }
  return "";
}

int main(int, char**) {
  RawArrayOf<double> rawdouble(Identity::none(), 10);
  for (int i = 0;  i < 10;  i++) {
    *rawdouble.borrow(i) = 1.1*i;
  }

  Index32 offsetsA(6);
  offsetsA.ptr().get()[0] = 0;
  offsetsA.ptr().get()[1] = 3;
  offsetsA.ptr().get()[2] = 3;
  offsetsA.ptr().get()[3] = 5;
  offsetsA.ptr().get()[4] = 6;
  offsetsA.ptr().get()[5] = 10;

  Index32 offsetsB(5);
  offsetsB.ptr().get()[0] = 0;
  offsetsB.ptr().get()[1] = 3;
  offsetsB.ptr().get()[2] = 4;
  offsetsB.ptr().get()[3] = 4;
  offsetsB.ptr().get()[4] = 5;

  std::shared_ptr<Content> content(new RawArrayOf<double>(rawdouble));
  if (tostring(content) != "[0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]") {
    return -1;
  }
  if (tostring(content.get()->getitem_range(1, -1)) != "[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]") {
    return -1;
  }

  std::shared_ptr<Content> listA(new ListOffsetArray32(Identity::none(), offsetsA, content));
  if (tostring(listA) != "[[0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]") {
    return -1;
  }
  if (tostring(listA.get()->getitem_range(1, -1)) != "[[], [3.3, 4.4], [5.5]]") {
    return -1;
  }

  std::shared_ptr<Content> listB(new ListOffsetArray32(Identity::none(), offsetsB, listA));
  if (tostring(listB) != "[[[0, 1.1, 2.2], [], [3.3, 4.4]], [[5.5]], [], [[6.6, 7.7, 8.8, 9.9]]]") {
    return -1;
  }
  if (tostring(listB.get()->getitem_range(1, -1)) != "[[[5.5]], []]") {
    return -1;
  }

  return 0;
}
