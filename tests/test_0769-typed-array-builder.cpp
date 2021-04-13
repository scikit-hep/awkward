// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include <iostream>
#include <vector>

#include "awkward/Content.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/forth/ForthInputBuffer.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"

namespace ak = awkward;

int main(int, char**) {
  auto options = ak::ArrayBuilderOptions(8, 2);

  const ak::FormPtr empty_form = std::make_shared<ak::EmptyForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node0"));

  const ak::FormPtr numpy_form_int64 = std::make_shared<ak::NumpyForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node0"),
    std::vector<int64_t>(),
    ak::util::dtype_to_itemsize(ak::util::dtype::int64),
    ak::util::dtype_to_format(ak::util::dtype::int64),
    ak::util::dtype::int64);

  const ak::FormPtr numpy_form_float64 = std::make_shared<ak::NumpyForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node1"),
    std::vector<int64_t>(),
    ak::util::dtype_to_itemsize(ak::util::dtype::float64),
    ak::util::dtype_to_format(ak::util::dtype::float64),
    ak::util::dtype::float64);

  ak::FormPtr list_offset_form = std::make_shared<ak::ListOffsetForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node2"),
    ak::Index::Form::i64,
    numpy_form_float64);

  ak::FormPtr list_form = std::make_shared<ak::ListForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node3"),
    ak::Index::Form::i64,
    ak::Index::Form::i64,
    numpy_form_float64);

  ak::FormPtr record_form = std::make_shared<ak::RecordForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node4"),
    std::make_shared<ak::util::RecordLookup>(ak::util::RecordLookup({ "x", "y" })),
    std::vector<ak::FormPtr>({ numpy_form_int64, list_offset_form }));

  {
    // create builder
    ak::TypedArrayBuilder myarray(empty_form, options);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(numpy_form_float64, options);

    myarray.float64(999.999);
    myarray.float64(-999.999);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(list_offset_form, options);

    myarray.begin_list();
    myarray.float64(1.1);
    myarray.float64(2.2);
    myarray.end_list();

    myarray.begin_list();
    myarray.float64(3.3);
    myarray.float64(4.4);
    myarray.float64(5.5);
    myarray.float64(6.6);
    myarray.end_list();

    myarray.begin_list();
    myarray.float64(7.7);
    myarray.float64(8.8);
    myarray.float64(9.9);
    myarray.float64(10.1);
    myarray.end_list();

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(record_form, options);

    myarray.int64(1);
    myarray.begin_list();
    myarray.float64(1.1);
    myarray.float64(2.2);
    myarray.end_list();

    myarray.int64(2);
    myarray.begin_list();
    myarray.float64(3.3);
    myarray.float64(4.4);
    myarray.float64(5.5);
    myarray.float64(6.6);
    myarray.end_list();

    myarray.int64(3);
    myarray.begin_list();
    myarray.float64(7.7);
    myarray.float64(8.8);
    myarray.float64(9.9);
    myarray.float64(10.1);
    myarray.end_list();

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  return 0;
}
