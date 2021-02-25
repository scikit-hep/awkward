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
#include "awkward/builder/TypedArrayBuilder.h"

namespace ak = awkward;

int main(int, char**) {
  auto options = ak::ArrayBuilderOptions(1024, 2);

  const ak::FormPtr empty_form = std::make_shared<ak::EmptyForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node0"));

  const ak::FormPtr numpy_form = std::make_shared<ak::NumpyForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node1"),
    std::vector<int64_t>(),
    8,
    "d",
    ak::util::dtype::float64);

  ak::FormPtr list_offset_form = std::make_shared<ak::ListOffsetForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node2"),
    ak::Index::Form::i64,
    numpy_form);

  ak::FormPtr list_form = std::make_shared<ak::ListForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node3"),
    ak::Index::Form::i64,
    ak::Index::Form::i64,
    numpy_form);

  ak::FormPtr record_form = std::make_shared<ak::RecordForm>(
    false,
    ak::util::Parameters(),
    std::make_shared<std::string>("node4"),
    nullptr,
    std::vector<ak::FormPtr>());

  {
    // create builder
    ak::TypedArrayBuilder myarray(empty_form, options);
    const std::shared_ptr<ak::ForthMachine32> vm =
      std::make_shared<ak::ForthMachine32>(std::string("begin\n")
                                          .append("pause\n")
                                          .append("again\n"));

    myarray.connect(vm);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(numpy_form, options);

    const std::shared_ptr<ak::ForthMachine32> vm =
      std::make_shared<ak::ForthMachine32>(std::string("input data\n")
                                          .append("output part0-node1-data float64\n")
                                          .append("\n")
                                          .append(": node1-float64\n")
                                          .append("1 = if\n")
                                          .append("0 data seek\n")
                                          .append("data d-> part0-node1-data\n")
                                          .append("else\n")
                                          .append("halt\n")
                                          .append("then\n")
                                          .append(";\n")
                                          .append("\n")
                                          .append("begin\n")
                                          .append("node1-float64 pause\n")
                                          .append("again\n"));

    myarray.connect(vm);

    myarray.real(999.999);
    myarray.real(-999.999);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(list_offset_form, options);

    const std::shared_ptr<ak::ForthMachine32> vm =
      std::make_shared<ak::ForthMachine32>(std::string("input data\n")
                                          .append("output part0-node1-data float64\n")
                                          .append("output part0-node2-offsets int64\n")
                                          .append("\n")
                                          .append(": node1-float64\n")
                                          .append("1 = if\n") // real = 1
                                          .append("0 data seek\n")
                                          .append("data d-> part0-node1-data\n")
                                          .append("else\n")
                                          .append("halt\n")
                                          .append("then\n")
                                          .append(";\n")
                                          .append(": node2-list\n")
                                          .append("2 <> if\n") // beginlist = 2
                                          .append("halt\n")
                                          .append("then\n")
                                          .append("\n")
                                          .append("0\n")
                                          .append("begin\n")
                                          .append("pause\n")
                                          .append("dup 3 = if\n") // endlist = 3
                                          .append("drop\n")
                                          .append("part0-node2-offsets +<- stack\n")
                                          .append("exit\n")
                                          .append("else\n")
                                          .append("node1-float64\n")
                                          .append("1+\n")
                                          .append("then\n")
                                          .append("again\n")
                                          .append(";\n")
                                          .append("\n")
                                          .append("begin\n")
                                          .append("pause\n")
                                          .append("node2-list\n")
                                          .append("+1\n")
                                          .append("again\n"));

    myarray.connect(vm);

    myarray.beginlist();
    myarray.real(1.1);
    myarray.real(2.2);
    myarray.endlist();

    myarray.beginlist();
    myarray.real(3.3);
    myarray.real(4.4);
    myarray.real(5.5);
    myarray.real(6.6);
    myarray.endlist();

    myarray.beginlist();
    myarray.real(7.7);
    myarray.real(8.8);
    myarray.real(9.9);
    myarray.real(10.1);
    myarray.endlist();

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  // {
  //   // create another builder
  //   ak::TypedArrayBuilder myarray(list_form, options);
  //
  //   // take a snapshot
  //   std::shared_ptr<ak::Content> array = myarray.snapshot();
  //
  //   std::cout << array.get()->tostring() << "\n";
  // }
  // {
  //   ak::FormKey numpy_bool_form_key = std::make_shared<std::string>("one");
  //
  //   const ak::FormPtr numpy_bool_form = std::make_shared<ak::NumpyForm>(
  //     false,
  //     ak::util::Parameters(),
  //     numpy_bool_form_key,
  //     std::vector<int64_t>(),
  //     ak::util::dtype_to_itemsize(ak::util::dtype::boolean),
  //     ak::util::dtype_to_format(ak::util::dtype::boolean),
  //     ak::util::dtype::boolean);
  //
  //   ak::FormKey numpy_int_form_key = std::make_shared<std::string>("two");
  //
  //   const ak::FormPtr numpy_int_form = std::make_shared<ak::NumpyForm>(
  //     false,
  //     ak::util::Parameters(),
  //     numpy_int_form_key,
  //     std::vector<int64_t>(),
  //     ak::util::dtype_to_itemsize(ak::util::dtype::int64),
  //     ak::util::dtype_to_format(ak::util::dtype::int64),
  //     ak::util::dtype::int64);
  //
  //   // create another builder
  //   ak::TypedArrayBuilder myarray(record_form, options);
  //   //
  //   // myarray.field_check("two");
  //   // myarray.integer(999);
  //   // myarray.integer(-999);
  //   //
  //   // myarray.apply(numpy_bool_form, booleans, 3);
  //   // myarray.apply(numpy_bool_form, booleans, 5);
  //   //
  //   // myarray.field_check("one");
  //   // myarray.boolean(true);
  //   // myarray.boolean(false);
  //
  //   // The following will throw an exception
  //   // myarray.field_check("three");
  //
  //   // take a snapshot
  //   std::shared_ptr<ak::Content> array = myarray.snapshot();
  //
  //   std::cout << array.get()->tostring() << "\n";
  // }
  return 0;
}
