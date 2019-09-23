// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <iostream>

#include "awkward/Identity.h"
#include "awkward/RawArray.h"
#include "awkward/Slice.h"

using namespace awkward;

void rawarray() {
  RawArrayOf<float> data(Identity::none(), 4);
  *data.borrow(0) = 0.0f;
  *data.borrow(1) = 1.1f;
  *data.borrow(2) = 2.2f;
  *data.borrow(3) = 3.3f;
  assert(*dynamic_cast<RawArrayOf<float>*>(data.get(1).get())->borrow() == 1.1f);
  assert(*dynamic_cast<RawArrayOf<float>*>(data.slice(1, 3).get())->borrow(0) == 1.1f);
  assert(*dynamic_cast<RawArrayOf<float>*>(data.slice(1, 3).get())->borrow(1) == 2.2f);
}

void slices() {
  RawArrayOf<float> data(Identity::none(), 9);
  *data.borrow(0) = 0.0f;
  *data.borrow(1) = 1.1f;
  *data.borrow(2) = 2.2f;
  *data.borrow(3) = 3.3f;
  *data.borrow(4) = 4.4f;
  *data.borrow(5) = 5.5f;
  *data.borrow(6) = 6.6f;
  *data.borrow(7) = 7.7f;
  *data.borrow(8) = 8.8f;
  *data.borrow(9) = 9.9f;

  // Slice none(std::vector<std::shared_ptr<SliceItem>>(), true);
  // assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(none).get())->borrow() == 0.0f);

  Slice at1(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceAt(1)) }), true);
  assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(at1).get())->borrow() == 1.1f);
  Slice at2(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceAt(2)) }), true);
  assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(at2).get())->borrow() == 2.2f);

  Slice range1(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceRange(1, 3, Slice::none())) }));
  assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(range1).get())->borrow(0) == 1.1f);

  Slice range2(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceRange(Slice::none(), 4, 1)) }));
  assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(range2).get())->borrow(0) == 0.0f);

  // Slice range3(std::vector<std::shared_ptr<SliceItem>>({ std::shared_ptr<SliceItem>(new SliceRange(1, Slice::none(), 2)) }));
  // assert(*dynamic_cast<RawArrayOf<float>*>(data.getitem(range3).get())->borrow(1) == 3.3f);

}

int main(int, char**) {
  rawarray();
  slices();
}
