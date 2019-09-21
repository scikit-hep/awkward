// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <iostream>

#include "awkward/Identity.h"
#include "awkward/RawArray.h"
#include "awkward/Slice.h"

using namespace awkward;

// void rawarray() {
//   RawArrayOf<float> data(Identity::none(), 4);
//   *data.borrow(0) = 0.0f;
//   *data.borrow(1) = 1.1f;
//   *data.borrow(2) = 2.2f;
//   *data.borrow(3) = 3.3f;
//   assert(*dynamic_cast<RawArrayOf<float>*>(data.get(1).get())->borrow(0) == 1.1f);
//   assert(*dynamic_cast<RawArrayOf<float>*>(data.slice(1, 3).get())->borrow(0) == 1.1f);
// }
//
// void slices() {
//   Slice slice = Slice().with(SliceAt(1))
//                        .with(SliceStartStop(1, 3))
//                        .with(SliceStartStop(Slice::none(), Slice::none()))
//                        .with(SliceStartStopStep(Slice::none(), Slice::none(), 2))
//                        .with(SliceByteMask(Index8(10)))
//                        .with(SliceIndex32(Index32(15)))
//                        .with(SliceIndex64(Index64(20)))
//                        .with(SliceEllipsis())
//                        .with(SliceNewAxis());
//   assert(slice.length() == 9);
// }

int main(int, char**) {
  // rawarray();
  // slices();
}
