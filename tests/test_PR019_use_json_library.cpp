// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Slice.h"
#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/FillableOptions.h"

namespace ak = awkward;

int main(int, char**) {
  std::vector<std::vector<std::vector<double>>> vector =
    {{{0.0, 1.1, 2.2}, {}, {3.3, 4.4}}, {{5.5}}, {}, {{6.6, 7.7, 8.8, 9.9}}};

  ak::FillableArray builder(ak::FillableOptions(1024, 2.0));
  for (auto x : vector) builder.fill(x);
  std::shared_ptr<ak::Content> array = builder.snapshot();

  ak::Slice slice;
  slice.append(ak::SliceRange(ak::Slice::none(), ak::Slice::none(), -1));
  slice.append(ak::SliceRange(ak::Slice::none(), ak::Slice::none(), 2));
  slice.append(ak::SliceRange(1, ak::Slice::none(), ak::Slice::none()));

  if (array.get()->getitem(slice).get()->tojson(false, 1) !=
         "[[[7.7,8.8,9.9]],[],[[]],[[1.1,2.2],[4.4]]]")
    return -1;
  return 0;
}
