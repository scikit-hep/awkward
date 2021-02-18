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

#include "awkward/builder/TypedArrayBuilder.h"

namespace ak = awkward;

int main(int, char**) {

  const ak::FormPtr empty_form = std::make_shared<ak::EmptyForm>(
    false,
    ak::util::Parameters(),
    ak::FormKey(nullptr));

  const ak::FormPtr numpy_form = std::make_shared<ak::NumpyForm>(
    false,
    ak::util::Parameters(),
    ak::FormKey(nullptr),
    std::vector<int64_t>(),
    8,
    "d",
    ak::util::dtype::float64);

  ak::FormPtr list_offset_form = std::make_shared<ak::ListOffsetForm>(
    false,
    ak::util::Parameters(),
    ak::FormKey(nullptr),
    ak::Index::Form::i64,
    numpy_form);

  ak::FormPtr list_form = std::make_shared<ak::ListForm>(
    false,
    ak::util::Parameters(),
    ak::FormKey(nullptr),
    ak::Index::Form::i64,
    ak::Index::Form::i64,
    numpy_form);

  ak::FormPtr record_form = std::make_shared<ak::RecordForm>(
    false,
    ak::util::Parameters(),
    ak::FormKey(nullptr),
    nullptr,
    std::vector<ak::FormPtr>());

  // FIXME: this should come from ak::ForthMachine
  const std::shared_ptr<void> ptr = ak::kernel::malloc<void>(
    ak::kernel::lib::cpu, 1024);

  reinterpret_cast<double*>(ptr.get())[0] = 1.1;
  reinterpret_cast<double*>(ptr.get())[1] = 2.2;
  reinterpret_cast<double*>(ptr.get())[2] = 3.3;
  reinterpret_cast<double*>(ptr.get())[3] = 4.4;
  reinterpret_cast<double*>(ptr.get())[4] = 5.5;
  reinterpret_cast<double*>(ptr.get())[5] = 6.6;
  reinterpret_cast<double*>(ptr.get())[6] = 7.7;
  reinterpret_cast<double*>(ptr.get())[7] = 8.8;
  reinterpret_cast<double*>(ptr.get())[8] = 9.9;
  reinterpret_cast<double*>(ptr.get())[9] = 10.10;

  const std::shared_ptr<void> index = ak::kernel::malloc<void>(
    ak::kernel::lib::cpu, 1024);

  reinterpret_cast<int64_t*>(index.get())[0] = 0;
  reinterpret_cast<int64_t*>(index.get())[1] = 3;
  reinterpret_cast<int64_t*>(index.get())[2] = 3;
  reinterpret_cast<int64_t*>(index.get())[3] = 5;
  reinterpret_cast<int64_t*>(index.get())[4] = 8;
  reinterpret_cast<int64_t*>(index.get())[5] = 9;
  reinterpret_cast<int64_t*>(index.get())[6] = 100;
  reinterpret_cast<int64_t*>(index.get())[7] = 300;
  reinterpret_cast<int64_t*>(index.get())[8] = 300;
  reinterpret_cast<int64_t*>(index.get())[9] = 500;
  reinterpret_cast<int64_t*>(index.get())[10] = 800;
  reinterpret_cast<int64_t*>(index.get())[11] = 900;

  const std::shared_ptr<void> starts_stops = ak::kernel::malloc<void>(
    ak::kernel::lib::cpu, 1024);

  reinterpret_cast<int64_t*>(starts_stops.get())[0] = 0;
  reinterpret_cast<int64_t*>(starts_stops.get())[1] = 3;
  reinterpret_cast<int64_t*>(starts_stops.get())[2] = 7;
  reinterpret_cast<int64_t*>(starts_stops.get())[3] = 2;
  reinterpret_cast<int64_t*>(starts_stops.get())[4] = 6;
  reinterpret_cast<int64_t*>(starts_stops.get())[5] = 9;

  const std::shared_ptr<void> booleans = ak::kernel::malloc<void>(
    ak::kernel::lib::cpu, 1024);

  reinterpret_cast<bool*>(booleans.get())[0] = true;
  reinterpret_cast<bool*>(booleans.get())[1] = false;
  reinterpret_cast<bool*>(booleans.get())[2] = true;
  reinterpret_cast<bool*>(booleans.get())[3] = true;
  reinterpret_cast<bool*>(booleans.get())[4] = true;
  reinterpret_cast<bool*>(booleans.get())[5] = false;


  // auto input = ak::ForthInputBuffer(ptr, 0, 1024);
  // ak::util::ForthError err;
  // std::cout << "input buffer length is " << input.len() << "\n";
  // auto data = input.read(1024, err);

  {
    // create builder
    ak::TypedArrayBuilder myarray(1024);
    myarray.apply(empty_form, ptr, 20);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(1024);
    myarray.apply(numpy_form, ptr, 10);
    myarray.real(999.999);
    myarray.real(-999.999);

    const std::shared_ptr<void> ptr2 = ak::kernel::malloc<void>(
      ak::kernel::lib::cpu, 1024);

    reinterpret_cast<double*>(ptr2.get())[0] = -1.1;
    reinterpret_cast<double*>(ptr2.get())[1] = -2.2;
    reinterpret_cast<double*>(ptr2.get())[2] = -3.3;
    reinterpret_cast<double*>(ptr2.get())[3] = -4.4;
    reinterpret_cast<double*>(ptr2.get())[4] = -5.5;
    reinterpret_cast<double*>(ptr2.get())[5] = -6.6;
    reinterpret_cast<double*>(ptr2.get())[6] = -7.7;
    reinterpret_cast<double*>(ptr2.get())[7] = -8.8;
    reinterpret_cast<double*>(ptr2.get())[8] = -9.9;
    reinterpret_cast<double*>(ptr2.get())[9] = -10.10;

    myarray.apply(numpy_form, ptr2, 5);
    myarray.apply(numpy_form, ptr2, 5);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(1024);
    myarray.apply(list_offset_form, index, 6);
    myarray.apply(numpy_form, ptr, 10);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    // create another builder
    ak::TypedArrayBuilder myarray(1024);
    myarray.apply(list_form, starts_stops, 3);
    myarray.apply(numpy_form, ptr, 10);

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  {
    ak::FormKey numpy_bool_form_key = std::make_shared<std::string>("one");

    const ak::FormPtr numpy_bool_form = std::make_shared<ak::NumpyForm>(
      false,
      ak::util::Parameters(),
      numpy_bool_form_key,
      std::vector<int64_t>(),
      ak::util::dtype_to_itemsize(ak::util::dtype::boolean),
      ak::util::dtype_to_format(ak::util::dtype::boolean),
      ak::util::dtype::boolean);

    ak::FormKey numpy_int_form_key = std::make_shared<std::string>("two");

    const ak::FormPtr numpy_int_form = std::make_shared<ak::NumpyForm>(
      false,
      ak::util::Parameters(),
      numpy_int_form_key,
      std::vector<int64_t>(),
      ak::util::dtype_to_itemsize(ak::util::dtype::int64),
      ak::util::dtype_to_format(ak::util::dtype::int64),
      ak::util::dtype::int64);

    // create another builder
    ak::TypedArrayBuilder myarray(1024);
    myarray.apply(record_form);

    // FIXME: if this is allowed, e.g. a data buffer is a nullptr
    // an extra check is needed downstream
    myarray.apply(numpy_bool_form);

    myarray.apply(numpy_int_form, index, 5);
    myarray.apply(numpy_int_form, index, 5);
    myarray.apply(numpy_int_form, index, 2);

    myarray.field_check("two");
    myarray.integer(999);
    myarray.integer(-999);

    myarray.apply(numpy_bool_form, booleans, 3);
    myarray.apply(numpy_bool_form, booleans, 5);

    myarray.field_check("one");
    myarray.boolean(true);
    myarray.boolean(false);

    // The following will throw an exception
    // myarray.field_check("three");

    // take a snapshot
    std::shared_ptr<ak::Content> array = myarray.snapshot();

    std::cout << array.get()->tostring() << "\n";
  }
  return 0;
}
