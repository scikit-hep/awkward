// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <iostream>

#include "awkward/Identity.h"
#include "awkward/RawArray.h"

using namespace awkward;

int main(int, char**) {
  RawArrayOf<float> data(Identity::none(), 4);
  *data.borrow(0) = 0.0;
  *data.borrow(1) = 1.1;
  *data.borrow(2) = 2.2;
  *data.borrow(3) = 3.3;

  return 0;
}
