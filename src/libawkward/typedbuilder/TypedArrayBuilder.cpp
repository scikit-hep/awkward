// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/type/Type.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"

#include "awkward/typedbuilder/BitMaskedArrayBuilder.h"
#include "awkward/typedbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/typedbuilder/EmptyArrayBuilder.h"
#include "awkward/typedbuilder/IndexedArrayBuilder.h"
#include "awkward/typedbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/typedbuilder/ListArrayBuilder.h"
#include "awkward/typedbuilder/ListOffsetArrayBuilder.h"
#include "awkward/typedbuilder/NumpyArrayBuilder.h"
#include "awkward/typedbuilder/RecordArrayBuilder.h"
#include "awkward/typedbuilder/RegularArrayBuilder.h"
#include "awkward/typedbuilder/UnionArrayBuilder.h"
#include "awkward/typedbuilder/UnmaskedArrayBuilder.h"
#include "awkward/typedbuilder/VirtualArrayBuilder.h"
#include "awkward/typedbuilder/UnknownFormBuilder.h"


namespace awkward {

  const std::string
  index_form_to_name(Index::Form form) {
    switch (form) {
    case Index::Form::i8:
      return "int8";
    case Index::Form::u8:
      return "uint8";
    case Index::Form::i32:
      return "int32";
    case Index::Form::u32:
      return "uint32";
    case Index::Form::i64:
      return "int64";
    default:
      throw std::runtime_error(
        std::string("unrecognized Index::Form ") + FILENAME(__LINE__));
    }
  }

  const std::string
  dtype_to_state(util::dtype dt) {
    switch (dt) {
    case util::dtype::boolean:
      return std::to_string(static_cast<utype>(state::boolean));
    case util::dtype::int8:
      return std::to_string(static_cast<utype>(state::int8));
    case util::dtype::int16:
      return std::to_string(static_cast<utype>(state::int16));
    case util::dtype::int32:
      return std::to_string(static_cast<utype>(state::int32));
    case util::dtype::int64:
      return std::to_string(static_cast<utype>(state::int64));
    case util::dtype::uint8:
      return std::to_string(static_cast<utype>(state::uint8));
    case util::dtype::uint16:
      return std::to_string(static_cast<utype>(state::uint16));
    case util::dtype::uint32:
      return std::to_string(static_cast<utype>(state::uint32));
    case util::dtype::uint64:
      return std::to_string(static_cast<utype>(state::uint64));
    case util::dtype::float16:
      return std::to_string(static_cast<utype>(state::float16));
    case util::dtype::float32:
      return std::to_string(static_cast<utype>(state::float32));
    case util::dtype::float64:
      return std::to_string(static_cast<utype>(state::float64));
    case util::dtype::float128:
      return std::to_string(static_cast<utype>(state::float128));
    case util::dtype::complex64:
      return std::to_string(static_cast<utype>(state::complex64));
    case util::dtype::complex128:
      return std::to_string(static_cast<utype>(state::complex128));
    case util::dtype::complex256:
      return std::to_string(static_cast<utype>(state::complex256));
      // case datetime64:
      //   return static_cast<utype>(state::datetime64);
      // case timedelta64:
      //   return static_cast<utype>(state::timedelta64);
    default:
      throw std::runtime_error(
        std::string("unrecognized util::dtype ") + FILENAME(__LINE__));
    }
  };

  const std::string
  dtype_to_vm_format(util::dtype dt) {
    switch (dt) {
    case util::dtype::boolean:
    case util::dtype::int8:
    case util::dtype::int16:
    case util::dtype::int32:
    case util::dtype::int64:
    case util::dtype::uint8:
    case util::dtype::uint16:
    case util::dtype::uint32:
    case util::dtype::uint64:
      return "q";
    case util::dtype::float16:
    case util::dtype::float32:
    case util::dtype::float64:
    case util::dtype::float128:
    case util::dtype::complex64:
    case util::dtype::complex128:
    case util::dtype::complex256:
 // case datetime64:
 // case timedelta64:
      return "d";
    default:
      throw std::runtime_error(
        std::string("unrecognized util::dtype ") + FILENAME(__LINE__));
    }
  };

  TypedArrayBuilder::TypedArrayBuilder(const FormPtr& form,
                                       const ArrayBuilderOptions& options)
    : initial_(options.initial()),
      builder_(formBuilderFromA(form)),
      vm_input_data_("data"),
      vm_source_() {
    vm_source_.append("input ")
      .append(vm_input_data_)
      .append("\n");

    vm_source_.append(builder_.get()->vm_output()).append("\n");
    vm_source_.append(builder_.get()->vm_func()).append("\n");
    vm_source_.append(builder_.get()->vm_from_stack()).append("\n");

    vm_source_.append("0\n").append("begin\n")
    .append("pause\n")
    .append(builder_.get()->vm_func_name())
    .append("\n")
    .append("1+\n")
    .append("again\n");
  }

  FormBuilderPtr
  TypedArrayBuilder::formBuilderFromA(const FormPtr& form) {
    if (auto const& downcasted_form = std::dynamic_pointer_cast<BitMaskedForm>(form)) {
      return std::make_shared<BitMaskedArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<ByteMaskedForm>(form)) {
      return std::make_shared<ByteMaskedArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<EmptyForm>(form)) {
      return std::make_shared<EmptyArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<IndexedForm>(form)) {
      return std::make_shared<IndexedArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<IndexedOptionForm>(form)) {
      return std::make_shared<IndexedOptionArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<ListForm>(form)) {
      return std::make_shared<ListArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<ListOffsetForm>(form)) {
      return std::make_shared<ListOffsetArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<NumpyForm>(form)) {
      return std::make_shared<NumpyArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<RecordForm>(form)) {
      return std::make_shared<RecordArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<RegularForm>(form)) {
      return std::make_shared<RegularArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<UnionForm>(form)) {
      return std::make_shared<UnionArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<UnmaskedForm>(form)) {
      return std::make_shared<UnmaskedArrayBuilder>(downcasted_form);
    }
    else if (auto const& downcasted_form = std::dynamic_pointer_cast<VirtualForm>(form)) {
      return std::make_shared<VirtualArrayBuilder>(downcasted_form);
    }
    else {
      return std::make_shared<UnknownFormBuilder>(form);
    }
  }

  void
  TypedArrayBuilder::connect(const std::shared_ptr<ForthMachine32>& vm) {
    length_ = 8;
    vm_ = vm;
    std::shared_ptr<void> ptr(
      kernel::malloc<void>(kernel::lib::cpu, 8*sizeof(uint8_t)));

    vm_inputs_map_[vm_input_data_] = std::make_shared<ForthInputBuffer>(ptr, 0, length_);
    vm.get()->run(vm_inputs_map_);
  }

  void
  TypedArrayBuilder::debug_step() const {
    std::cout << "stack ";
    for (auto const& i : vm_.get()->stack()) {
      std::cout << i << ", ";
    }
    std::cout << "\n";
    for (auto const& i : vm_.get()->outputs()) {
      std::cout << i.first << " : ";
      std::cout << i.second.get()->toNumpyArray().get()->tostring();
      std::cout << "\n";
    }
  }

  const FormPtr
  TypedArrayBuilder::form() const {
    return builder_.get()->form();
  }

  const std::string
  TypedArrayBuilder::to_vm() const {
    return vm_source_;
  }

  const std::string
  TypedArrayBuilder::tostring() const {
    util::TypeStrs typestrs;
    typestrs["char"] = "char";
    typestrs["string"] = "string";
    std::stringstream out;
    out << "<TypedArrayBuilder length=\"" << length() << "\" type=\""
        << type(typestrs).get()->tostring() << "\"/>";
    return out.str();
  }

  int64_t
  TypedArrayBuilder::length() const {
    return length_;
  }

  void
  TypedArrayBuilder::clear() {
    if (builder_ != nullptr) {
      throw std::runtime_error(
        std::string("FormBuilder 'clear' is not implemented yet")
        + FILENAME(__LINE__));
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
  }

  const TypePtr
  TypedArrayBuilder::type(const util::TypeStrs& typestrs) const {
    return builder_.get()->snapshot(vm_.get()->outputs()).get()->type(typestrs);
  }

  const ContentPtr
  TypedArrayBuilder::snapshot() const {
    return builder_.get()->snapshot(vm_.get()->outputs());
  }

  const ContentPtr
  TypedArrayBuilder::getitem_at(int64_t at) const {
    return snapshot().get()->getitem_at(at);
  }

  const ContentPtr
  TypedArrayBuilder::getitem_range(int64_t start, int64_t stop) const {
    return snapshot().get()->getitem_range(start, stop);
  }

  const ContentPtr
  TypedArrayBuilder::getitem_field(const std::string& key) const {
    return snapshot().get()->getitem_field(key);
  }

  const ContentPtr
  TypedArrayBuilder::getitem_fields(const std::vector<std::string>& keys) const {
    return snapshot().get()->getitem_fields(keys);
  }

  const ContentPtr
  TypedArrayBuilder::getitem(const Slice& where) const {
    return snapshot().get()->getitem(where);
  }

  void
  TypedArrayBuilder::null() {
    reinterpret_cast<int64_t*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = -1;
    vm_.get()->stack_push(static_cast<utype>(state::null));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::boolean(bool x) {
    reinterpret_cast<bool*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
    vm_.get()->stack_push(static_cast<utype>(state::boolean));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::integer(int64_t x) {
    reinterpret_cast<int64_t*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
    vm_.get()->stack_push(static_cast<utype>(state::int64));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::real(double x) {
    reinterpret_cast<double*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
    vm_.get()->stack_push(static_cast<utype>(state::float64));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::complex(std::complex<double> x) {
    reinterpret_cast<std::complex<double>*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = x;
    vm_.get()->stack_push(static_cast<utype>(state::complex128));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::bytestring(const char* x) {
    //builder_.get()->string(x, -1, no_encoding);
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'bytestring' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::bytestring(const char* x, int64_t length) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'bytestring' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::bytestring(const std::string& x) {
    bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  TypedArrayBuilder::string(const char* x) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'string' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::string(const char* x, int64_t length) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'string' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::string(const std::string& x) {
    string(x.c_str(), (int64_t)x.length());
  }

  void
  TypedArrayBuilder::beginlist() {
    vm_.get()->stack_push(static_cast<utype>(state::begin_list));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::endlist() {
    vm_.get()->stack_push(static_cast<utype>(state::end_list));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::begintuple(int64_t numfields) {
    vm_.get()->stack_push(static_cast<utype>(state::begin_tuple));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::index(int64_t index) {
    reinterpret_cast<int64_t*>(vm_inputs_map_[vm_input_data_]->ptr().get())[0] = index;
    vm_.get()->stack_push(static_cast<utype>(state::index));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::endtuple() {
    vm_.get()->stack_push(static_cast<utype>(state::end_tuple));
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::beginrecord() {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'beginrecord' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_fast(const char* name) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'beginrecord_fast' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_check(const char* name) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'beginrecord_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_check(const std::string& name) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'beginrecord_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_fast(const char* key) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'field_fast' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_check(const char* key) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'field_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_check(const std::string& key) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'field_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::endrecord() {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'endrecord' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::append(const ContentPtr& array, int64_t at) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'append' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::append_nowrap(const ContentPtr& array, int64_t at) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'append_nowrap' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::extend(const ContentPtr& array) {
    throw std::runtime_error(
      std::string("TypedArrayBuilder 'extend' is not implemented yet")
      + FILENAME(__LINE__));
  }

}
