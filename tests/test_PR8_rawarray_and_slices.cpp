// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <iostream>

#include "awkward/Identity.h"
#include "awkward/RawArray.h"
#include "awkward/Slice.h"

using namespace awkward;

void rawarray() {
  RawArrayOf<float> data(Identity::none(), 4);
  *data.borrow(0) = 0.0;
  *data.borrow(1) = 1.1;
  *data.borrow(2) = 2.2;
  *data.borrow(3) = 3.3;
  assert(*dynamic_cast<RawArrayOf<float>*>(data.get(1).get())->borrow(0) == 1.1f);
  assert(*dynamic_cast<RawArrayOf<float>*>(data.slice(1, 3).get())->borrow(0) == 1.1f);
}

void slices() {
  Slices slices;
  slices.append(Slice1(1));
  slices.append(Slice2(1, 3));

  std::cout << slices->tostring() << std::endl;
}

int main(int, char**) {
  rawarray();
  slices();
}
