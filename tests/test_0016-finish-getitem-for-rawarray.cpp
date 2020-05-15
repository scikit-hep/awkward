// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <iostream>

#include "awkward/Identities.h"
#include "awkward/array/RawArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
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
      out << tostring(listoffset->getitem_at_nowrap(i));
    }
    out << "]";
    return out.str();
  }
  else if (ListArray32* list = dynamic_cast<ListArray32*>(array.get())) {
    std::stringstream out;
    out << "[";
    for (int i = 0;  i < list->length();  i++) {
      if (i != 0) out << ", ";
      out << tostring(list->getitem_at_nowrap(i));
    }
    out << "]";
    return out.str();
  }
  return "";
}

Slice slice(SliceItem* s1) {
  Slice out;
  out.append(std::shared_ptr<SliceItem>(s1));
  out.become_sealed();
  return out;
}

Slice slice(SliceItem* s1, SliceItem* s2) {
  Slice out;
  out.append(std::shared_ptr<SliceItem>(s1));
  out.append(std::shared_ptr<SliceItem>(s2));
  out.become_sealed();
  return out;
}

Slice slice(SliceItem* s1, SliceItem* s2, SliceItem* s3) {
  Slice out;
  out.append(std::shared_ptr<SliceItem>(s1));
  out.append(std::shared_ptr<SliceItem>(s2));
  out.append(std::shared_ptr<SliceItem>(s3));
  out.become_sealed();
  return out;
}

int main(int, char**) {
  RawArrayOf<double> rawdouble(Identities::none(), util::Parameters(), 10);
  for (int i = 0;  i < 10;  i++) {
    *rawdouble.borrow(i) = 1.1*i;
  }

  Index32 offsetsA(6);
  offsetsA.setitem_at_nowrap(0, 0);
  offsetsA.setitem_at_nowrap(1, 3);
  offsetsA.setitem_at_nowrap(2, 3);
  offsetsA.setitem_at_nowrap(3, 5);
  offsetsA.setitem_at_nowrap(4, 6);
  offsetsA.setitem_at_nowrap(5, 10);

  Index32 offsetsB(5);
  offsetsB.setitem_at_nowrap(0, 0);
  offsetsB.setitem_at_nowrap(1, 3);
  offsetsB.setitem_at_nowrap(2, 4);
  offsetsB.setitem_at_nowrap(3, 4);
  offsetsB.setitem_at_nowrap(4, 5);

  std::shared_ptr<Content> content(new RawArrayOf<double>(rawdouble));
  if (tostring(content) != "[0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]") {
    return -1;
  }
  if (tostring(content.get()->getitem_range(1, -1)) != "[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]") {
    return -1;
  }
  if (tostring(content.get()->getitem(slice(new SliceAt(4)))) != "[4.4]") {
    return -1;
  }
  if (tostring(content.get()->getitem(slice(new SliceRange(6, Slice::none(), -1)))) != "[6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0]") {
    return -1;
  }
  if (tostring(content.get()->getitem(slice(new SliceRange(Slice::none(), Slice::none(), 2)))) != "[0, 2.2, 4.4, 6.6, 8.8]") {
    return -1;
  }
  if (tostring(content.get()->getitem(slice(new SliceRange(1, 9, 2)))) != "[1.1, 3.3, 5.5, 7.7]") {
    return -1;
  }

  Index64 array1(4);
  array1.setitem_at_nowrap(0, 2);
  array1.setitem_at_nowrap(1, 0);
  array1.setitem_at_nowrap(2, 0);
  array1.setitem_at_nowrap(3, -1);
  if (tostring(content.get()->getitem(slice(new SliceArray64(array1, std::vector<int64_t>({4}), std::vector<int64_t>({1}), false)))) != "[2.2, 0, 0, 9.9]") {
    return -1;
  }

  std::shared_ptr<Content> listA(new ListOffsetArray32(Identities::none(), util::Parameters(), offsetsA, content));
  if (tostring(listA) != "[[0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]") {
    return -1;
  }
  if (tostring(listA.get()->getitem_range(1, -1)) != "[[], [3.3, 4.4], [5.5]]") {
    return -1;
  }
  if (tostring(listA.get()->getitem(slice(new SliceAt(2)))) != "[3.3, 4.4]") {
    return -1;
  }
  if (tostring(listA.get()->getitem(slice(new SliceAt(2), new SliceAt(1)))) != "[4.4]") {
    return -1;
  }
  if (tostring(listA.get()->getitem(slice(new SliceRange(2, Slice::none(), Slice::none()), new SliceRange(Slice::none(), -1, Slice::none())))) != "[[3.3], [], [6.6, 7.7, 8.8]]") {
    return -1;
  }
  if (tostring(listA.get()->getitem(slice(new SliceRange(2, Slice::none(), Slice::none()), new SliceRange(Slice::none(), Slice::none(), -1)))) != "[[4.4, 3.3], [5.5], [9.9, 8.8, 7.7, 6.6]]") {
    return -1;
  }

  Index64 array2(4);
  array2.setitem_at_nowrap(0, 1);
  array2.setitem_at_nowrap(1, -1);
  array2.setitem_at_nowrap(2, 2);
  array2.setitem_at_nowrap(3, 0);
  if (tostring(listA.get()->getitem(slice(new SliceArray64(array1, std::vector<int64_t>({4}), std::vector<int64_t>({1}), false), new SliceArray64(array2, std::vector<int64_t>({4}), std::vector<int64_t>({1}), false)))) != "[4.4, 2.2, 2.2, 6.6]") {
    return -1;
  }

  std::shared_ptr<Content> listB(new ListOffsetArray32(Identities::none(), util::Parameters(), offsetsB, listA));
  if (tostring(listB) != "[[[0, 1.1, 2.2], [], [3.3, 4.4]], [[5.5]], [], [[6.6, 7.7, 8.8, 9.9]]]") {
    return -1;
  }
  if (tostring(listB.get()->getitem_range(1, -1)) != "[[[5.5]], []]") {
    return -1;
  }

  return 0;
}
