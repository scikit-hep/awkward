// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/builder/TypedBuilder.h"
#include <memory>

namespace ak = awkward;

int main() {
  auto options = ak::ArrayBuilderOptions(1024, 2.0);
  auto bool_builder = ak::BoolTypedBuilder(options, ak::GrowableBuffer<uint8_t>::empty(options));
  auto int_builder = ak::Int64TypedBuilder(options, ak::GrowableBuffer<int64_t>::empty(options));

  integer(bool_builder, 1);
  integer(int_builder, 1);

  boolean(bool_builder, true);
  boolean(int_builder, true);

  std::shared_ptr<ak::Content> bool_array = bool_builder.snapshot();
  std::shared_ptr<ak::Content> int_array = int_builder.snapshot();

  std::cout << bool_array.get()->tostring() << "\n";
  std::cout << int_array.get()->tostring() << "\n";

  auto myarray = ak::TypedArrayBuilder<uint8_t, ak::BoolTypedBuilder>(ak::ArrayBuilderOptions(1024, 2.0));

// for(int64_t i = 0; i < 1000000000; i ++) {
  myarray.boolean(true);
  myarray.boolean(false);
  myarray.boolean(true);
  myarray.boolean(false);
  myarray.boolean(true);
  myarray.boolean(false);
  myarray.integer(1);
  myarray.real(1.1);
  myarray.null();
  myarray.boolean(false);
//}

  std::cout << myarray.snapshot().get()->tostring() << "\n";

  return 0;
}
