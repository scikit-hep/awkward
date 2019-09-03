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
  Slice slice;
  slice.append(SliceAt(1));
  slice.append(SliceStartStop(1, 3));
  slice.append(SliceStartStop(Slice::none(), Slice::none()));
  slice.append(SliceStartStopStep(Slice::none(), Slice::none(), 2));
  slice.append(SliceByteMask(Index8(10)));
  slice.append(SliceIndex32(Index32(15)));
  slice.append(SliceIndex64(Index64(20)));
  slice.append(SliceEllipsis());
  slice.append(SliceNewAxis());
  assert(slice.length() == 9);
}

int main(int, char**) {
  rawarray();
  slices();
}
