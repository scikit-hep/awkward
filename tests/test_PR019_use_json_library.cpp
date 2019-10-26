// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/fillable/FillableArray.h"
#include "awkward/fillable/FillableOptions.h"

namespace ak = awkward;

void build(ak::FillableArray& builder, double x) {
  builder.real(x);
}

template <typename T>
void build(ak::FillableArray& builder, const std::vector<T>& vector) {
  builder.beginlist();
  for (auto x : vector) build(builder, x);
  builder.endlist();
}

int main(int, char**) {
  std::vector<std::vector<std::vector<double>>> vector =
    {{{0.0, 1.1, 2.2}, {}, {3.3, 4.4}}, {{5.5}}, {}, {{6.6, 7.7, 8.8, 9.9}}};

  ak::FillableArray builder(ak::FillableOptions(1024, 2.0));
  for (auto x : vector) build(builder, x);

  if (builder.snapshot().get()->tojson(false, 1) !=
         "[[[0.0,1.1,2.2],[],[3.3,4.4]],[[5.5]],[],[[6.6,7.7,8.8,9.9]]]")
    return -1;

  return 0;
}
