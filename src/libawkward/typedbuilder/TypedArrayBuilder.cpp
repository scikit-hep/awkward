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

namespace awkward {
  FormBuilder::~FormBuilder() = default;

  /// FIXME: implement Form morfing
  /// for example, ListForm to ListOffsetForm
  ///
  FormBuilderPtr
  formBuilderFromA(const FormPtr& form) {
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
        std::string("unrecognized Index::Form") + FILENAME(__LINE__));
    }
  }

  enum class state : std::int32_t {
    int64 = 0,
    float64 = 1,
    begin_list = 2,
    end_list = 3,
    boolean = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    uint8 = 8,
    uint16 = 9,
    uint32 = 10,
    uint64 = 11,
    float16 = 12,
    float32 = 13,
    float128 = 14,
    complex64 = 15,
    complex128 = 16,
    complex256 = 17
  };
  using utype = std::underlying_type<state>::type;

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
      return std::to_string(-1);
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
      return "FIXME";
    }
  };

  TypedArrayBuilder::TypedArrayBuilder(const FormPtr& form,
                                       const ArrayBuilderOptions& options)
    : initial_(options.initial()),
      vm_input_data_("data"),
      vm_source_(),
      builder_(formBuilderFromA(form)) {
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
    return length_; // FIXME: builder_.get()->length();
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
    if (builder_ != nullptr) {
      return builder_.get()->snapshot(vm_.get()->outputs()).get()->type(typestrs);
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
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
    throw std::runtime_error(
      std::string("FormBuilder 'null' is not implemented yet")
      + FILENAME(__LINE__));
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
    throw std::runtime_error(
      std::string("FormBuilder 'complex' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::bytestring(const char* x) {
    //builder_.get()->string(x, -1, no_encoding);
    throw std::runtime_error(
      std::string("FormBuilder 'bytestring' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::bytestring(const char* x, int64_t length) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::bytestring(const std::string& x) {
    bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  TypedArrayBuilder::string(const char* x) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::string(const char* x, int64_t length) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
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
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::index(int64_t index) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::endtuple() {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord() {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_fast(const char* name) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_check(const char* name) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::beginrecord_check(const std::string& name) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_fast(const char* key) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_check(const char* key) {
    throw std::runtime_error(
      std::string("FormBuilder 'field_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::field_check(const std::string& key) {
    throw std::runtime_error(
      std::string("FormBuilder 'field_check' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::endrecord() {
    throw std::runtime_error(
      std::string("FormBuilder 'endrecord' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::append(const ContentPtr& array, int64_t at) {
    throw std::runtime_error(
      std::string("FormBuilder 'clear' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::append_nowrap(const ContentPtr& array, int64_t at) {
    throw std::runtime_error(
      std::string("FormBuilder 'append' is not implemented yet")
      + FILENAME(__LINE__));
  }

  void
  TypedArrayBuilder::extend(const ContentPtr& array) {
    throw std::runtime_error(
      std::string("FormBuilder 'extend' is not implemented yet")
      + FILENAME(__LINE__));
  }

  /// @brief
  BitMaskedArrayBuilder::BitMaskedArrayBuilder(const BitMaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-mask");

    vm_func_name_ = std::string(*form_key_).append("-").append("bitmask");

    vm_func_ = std::string(": ")
      .append(vm_func_name_).append("\n")
      .append(";").append("\n");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append("FIXME-type").append("\n");
  }

  const std::string
  BitMaskedArrayBuilder::classname() const {
    return "BitMaskedArrayBuilder";
  }

  const ContentPtr
  BitMaskedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtr out;
    int64_t length = 0; // FIXME
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      out = std::make_shared<BitMaskedArray>(Identities::none(),
                                             form_.get()->parameters(),
                                             search->second.get()->toIndexU8(),
                                             content_.get()->snapshot(outputs),
                                             form_.get()->valid_when(),
                                             length, // FIXME
                                             form_.get()->lsb_order());
    }
    return out;
  }

  const FormPtr
  BitMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  BitMaskedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  BitMaskedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-mask");
  }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtr out;
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      out = std::make_shared<ByteMaskedArray>(Identities::none(),
                                              form_.get()->parameters(),
                                              search->second.get()->toIndex8(),
                                              content_.get()->snapshot(outputs),
                                              form_.get()->valid_when());
    }
    return out;
  }

  const FormPtr
  ByteMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ByteMaskedArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_name() const {
    std::string out;
    out.append(*form_key_)
      .append("-")
      .append("bytemask");
    return out;
  }

  ///
  EmptyArrayBuilder::EmptyArrayBuilder(const EmptyFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

  const std::string
  EmptyArrayBuilder::classname() const {
    return "EmptyArrayBuilder";
  }

  const ContentPtr
  EmptyArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  const FormPtr
  EmptyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  EmptyArrayBuilder::vm_output() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func_name() const {
    return vm_empty_command_;
  }

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const IndexedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-index");
  }

  const std::string
  IndexedArrayBuilder::classname() const {
    return "IndexedArrayBuilder";
  }

  const ContentPtr
  IndexedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // if(content_ != nullptr) {
    //   Index64 index(reinterpret_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
    //   return std::make_shared<IndexedArray64>(Identities::none(),
    //                                           form_.get()->parameters(),
    //                                           index,
    //                                           content_.get()->snapshot(outputs));
    // }
    // else {
    //   throw std::invalid_argument(
    //     std::string("Form of a ") + classname()
    //     + std::string(" needs another Form as its content")
    //     + FILENAME(__LINE__));
    // }
    return nullptr;
  }

  const FormPtr
  IndexedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  IndexedArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  IndexedArrayBuilder::vm_func_name() const {
    return std::string (*form_key_)
      .append("-")
      .append("index");
  }

  ///
  IndexedOptionArrayBuilder::IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) { }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const ContentPtr
  IndexedOptionArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // if(content_ != nullptr) {
    //   Index64 index(reinterpret_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
    //   return std::make_shared<IndexedOptionArray64>(Identities::none(),
    //                                                 form_.get()->parameters(),
    //                                                 index,
    //                                                 content_.get()->snapshot(outputs));
    // }
    // else {
    //   throw std::invalid_argument(
    //     std::string("Form of a ") + classname()
    //     + std::string(" needs another Form as its content")
    //     + FILENAME(__LINE__));
    // }
    return nullptr;
  }

  const FormPtr
  IndexedOptionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-")
      .append("index");
  }

  ///
  ListArrayBuilder::ListArrayBuilder(const ListFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) { }

  const std::string
  ListArrayBuilder::classname() const {
    return "ListArrayBuilder";
  }

  const ContentPtr
  ListArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // if(content_ != nullptr) {
    //   Index64 starts(reinterpret_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
    //   Index64 stops(reinterpret_pointer_cast<int64_t>(data_), length_, length_, kernel::lib::cpu);
    //   return std::make_shared<ListArray64>(Identities::none(),
    //                                        form_.get()->parameters(),
    //                                        starts,
    //                                        stops,
    //                                        content_.get()->snapshot(outputs));
    // }
    // else {
    //   throw std::invalid_argument(
    //     std::string("Form of a ") + classname()
    //     + std::string(" needs another Form as its content")
    //     + FILENAME(__LINE__));
    // }
    return nullptr;
  }

  const FormPtr
  ListArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ListArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  ListArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  ListArrayBuilder::vm_func_name() const {
    std::string out;
    out.append(*form_key_)
      .append("-")
      .append("list");
    return out;
  }

  ///
  ListOffsetArrayBuilder::ListOffsetArrayBuilder(const ListOffsetFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-offsets");

    vm_func_name_ = std::string(*form_key_).append("-").append("list");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->offsets()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(std::to_string(static_cast<utype>(state::begin_list)))
      .append(" <> if").append("\n")
      .append("halt").append("\n")
      .append("then").append("\n")
      .append("\n")
      .append("0").append("\n")
      .append("begin").append("\n")
      .append("pause").append("\n")
      .append("dup ")
      .append(std::to_string(static_cast<utype>(state::end_list)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" +<- stack").append("\n")
      .append("exit").append("\n")
      .append("else").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("1+").append("\n")
      .append("then").append("\n")
      .append("again").append("\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");
  }

  const std::string
  ListOffsetArrayBuilder::classname() const {
    return "ListOffsetArrayBuilder";
  }

  const ContentPtr
  ListOffsetArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                 form_.get()->parameters(),
                                                 search->second.get()->toIndex64(),
                                                 content_.get()->snapshot(outputs));
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs offsets")
        + FILENAME(__LINE__));
  }

  const FormPtr
  ListOffsetArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ListOffsetArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const NumpyFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-data");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(dtype_to_name(form_.get()->dtype())).append("\n");

    vm_func_name_ = std::string(*form_key_)
      .append("-")
      .append(dtype_to_name(form_.get()->dtype()));

    vm_func_ = std::string(": ").append(vm_func_name()).append("\n")
      .append(dtype_to_state(form_.get()->dtype()))
      .append(" = if").append("\n")
      .append("0 data seek").append("\n")
      .append("data ").append(dtype_to_vm_format(form_.get()->dtype()))
      .append("-> ").append(vm_output_data_).append("\n")
      .append("else").append("\n")
      .append("halt").append("\n")
      .append("then").append("\n")
      .append(";").append("\n");
  }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const ContentPtr
  NumpyArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      return search->second.get()->toNumpyArray();
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs data")
        + FILENAME(__LINE__));
  }

  const FormPtr
  NumpyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  NumpyArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  NumpyArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  NumpyArrayBuilder::vm_from_stack() const {
    return std::string();
  }

  ///
  RecordArrayBuilder::RecordArrayBuilder(const RecordFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    for (auto const& content : form.get()->contents()) {
      contents_.push_back(formBuilderFromA(content));
    }

    for (auto const& content : contents_) {
      vm_output_.append(content.get()->vm_output());
    }
    for (auto const& content : contents_) {
      vm_from_stack_.append(content.get()->vm_from_stack());
    }

    vm_func_name_ = std::string(*form_key_).append("-record");

    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_);

    for (auto const& content : contents_) {
      vm_func_.append("\n    ").append(content.get()->vm_func_name())
        .append(" pause");
    }
    // Remove the last pause
    vm_func_.erase(vm_func_.end() - 6, vm_func_.end());
    vm_func_.append("\n;\n\n");
  }

  const std::string
  RecordArrayBuilder::classname() const {
    return "RecordArrayBuilder";
  }

  const ContentPtr
  RecordArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtrVec contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      contents.push_back(contents_[i].get()->snapshot(outputs));
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         form_.get()->parameters(),
                                         contents,
                                         form_.get()->recordlookup());
  }

  const FormPtr
  RecordArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RecordArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  RecordArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  RecordArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  RecordArrayBuilder::vm_from_stack() const {
    return vm_from_stack_;
  }

  ///
  RegularArrayBuilder::RegularArrayBuilder(const RegularFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

  const std::string
  RegularArrayBuilder::classname() const {
    return "RegularArrayBuilder";
  }

  const ContentPtr
  RegularArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtr out;
    if(content_ != nullptr) {
      int64_t length = 0; // outputs.len(); // FIXME
      out = std::make_shared<RegularArray>(Identities::none(),
                                           form_.get()->parameters(),
                                           content_.get()->snapshot(outputs),
                                           length); // FIXME
    }
    return out;
  }

  const FormPtr
  RegularArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RegularArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  RegularArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append("\n");
  }

  const std::string
  RegularArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-reg");
  }

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-tags");
 }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const ContentPtr
  UnionArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
      // return std::make_shared<UnionArray8_64>(Identities::none(),
      //                                         form_.get()->parameters());
  }

  const FormPtr
  UnionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnionArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  UnionArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(*form_key_)
      .append("-")
      .append("union\n");
  }

  const std::string
  UnionArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-union");
  }

  ///
  UnmaskedArrayBuilder::UnmaskedArrayBuilder(const UnmaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-mask");
  }

  const std::string
  UnmaskedArrayBuilder::classname() const {
    return "UnmaskedArrayBuilder";
  }

  const ContentPtr
  UnmaskedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // FIXME
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  const FormPtr
  UnmaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnmaskedArrayBuilder::vm_output() const {
    return std::string("output ")
      .append(vm_output_data_)
      .append("\n");
  }

  const std::string
  UnmaskedArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(*form_key_)
      .append("-")
      .append("unmasked\n");
  }

  const std::string
  UnmaskedArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-unmasked");
  }

  ///
  VirtualArrayBuilder::VirtualArrayBuilder(const VirtualFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

  const std::string
  VirtualArrayBuilder::classname() const {
    return "VirtualArrayBuilder";
  }

  const ContentPtr
  VirtualArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
  }

  const FormPtr
  VirtualArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  VirtualArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  VirtualArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  VirtualArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  ///
  UnknownFormBuilder::UnknownFormBuilder(const FormPtr& form)
    : form_(form) {}

  const std::string
  UnknownFormBuilder::classname() const {
    return "UnknownFormBuilder";
  }

  const ContentPtr
  UnknownFormBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    return nullptr;
  }

  const FormPtr
  UnknownFormBuilder::form() const {
    return form_;
  }

  const std::string
  UnknownFormBuilder::vm_output() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_func() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_func_name() const {
    return vm_empty_command_;
  }

}
