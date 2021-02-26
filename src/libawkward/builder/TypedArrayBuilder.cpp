// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/builder/TypedArrayBuilder.h"
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

  TypedArrayBuilder::TypedArrayBuilder(const FormPtr& form,
                                       const ArrayBuilderOptions& options)
    : initial_(options.initial()),
      builder_(formBuilderFromA(form)),
      vm_source_("input data\n") { }

  void
  TypedArrayBuilder::connect(const std::shared_ptr<ForthMachine32>& vm) {
    length_ = 8;
    vm_ = vm;
    std::shared_ptr<void> ptr(
      kernel::malloc<void>(kernel::lib::cpu, 8*sizeof(uint8_t)));

    vm_inputs_map_["data"] = std::make_shared<ForthInputBuffer>(ptr, 0, length_);
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
    // reinterpret_cast<bool*>(vm_inputs_map_["data"]->ptr_.get())[0] = x;
    // vm_.get()->stack_push(0);
    // vm_.get()->resume();
  }

  void
  TypedArrayBuilder::integer(int64_t x) {
    reinterpret_cast<int64_t*>(vm_inputs_map_["data"]->ptr_.get())[0] = x;
    vm_.get()->stack_push(0);
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::real(double x) {
    reinterpret_cast<double*>(vm_inputs_map_["data"]->ptr_.get())[0] = x;
    vm_.get()->stack_push(1);
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
    vm_.get()->stack_push(2);
    vm_.get()->resume();
  }

  void
  TypedArrayBuilder::endlist() {
    vm_.get()->stack_push(3);
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
      content_(formBuilderFromA(form.get()->content())) { }

  const std::string
  BitMaskedArrayBuilder::classname() const {
    return "BitMaskedArrayBuilder";
  }

  const ContentPtr
  BitMaskedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtr out;
    int64_t length = 0; // FIXME
    auto search = outputs.find(std::string("part0-").append(*form_key_).append("-mask"));
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

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) { }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    ContentPtr out;
    auto search = outputs.find(std::string("part0-").append(*form_key_).append("-mask"));
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

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const IndexedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())) { }

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

  ///
  ListArrayBuilder::ListArrayBuilder(const ListFormPtr& form,
                                     bool copyarrays)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())),
      copyarrays_(copyarrays) { }

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

  ///
  ListOffsetArrayBuilder::ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                                                 bool copyarrays)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(formBuilderFromA(form.get()->content())),
      copyarrays_(copyarrays) { }

  const std::string
  ListOffsetArrayBuilder::classname() const {
    return "ListOffsetArrayBuilder";
  }

  const ContentPtr
  ListOffsetArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    auto search = outputs.find(std::string("part0-").append(*form_key_).append("-offsets"));
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

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const NumpyFormPtr& form,
                                       bool copyarrays)
    : form_(form),
      form_key_(form.get()->form_key()),
      copyarrays_(copyarrays) { }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const ContentPtr
  NumpyArrayBuilder::snapshot(const ForthOtputBufferMap& outputs) const {
    auto search = outputs.find(std::string("part0-").append(*form_key_).append("-data"));
    if (search != outputs.end()) {
      return search->second.get()->toNumpyArray();
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs data")
        + FILENAME(__LINE__));
        
    // auto const& identities = Identities::none();
    //
    // // FIXME: check that length_ is always equal to
    // // Sum s = std::for_each(data_.begin(), data_.end(), Sum());
    //
    // std::vector<ssize_t> shape = { (ssize_t)length_ };
    // std::vector<ssize_t> strides = { (ssize_t)dtype_to_itemsize(form_.get()->dtype()) };
    //
    // if (copyarrays_) {
    //   Index64 bytepos(shape[0]);
    //   struct Error err1 = kernel::NumpyArray_contiguous_init_64(
    //     kernel::lib::cpu,   // DERIVE
    //     bytepos.data(),
    //     shape[0],
    //     strides[0]);
    //   util::handle_error(err1, classname(), identities.get());
    //
    //   std::shared_ptr<void> ptr(
    //     kernel::malloc<void>(kernel::lib::cpu, bytepos.length()*strides[0]));
    //
    //   // FIXME: move it to kernel?
    //   // nothing to be done if all data pointers hold the same memory address
    //   // otherwise, copy the data into a single contiguous memory
    //   //
    //   Index64 data_lens((int64_t)data_.size());
    //   const uint8_t* data_ptrs[data_.size()];
    //   int64_t indx = 0;
    //   for (auto const& it : data_) {
    //     data_ptrs[indx] = reinterpret_cast<uint8_t*>(it->ptr.get());
    //     data_lens.data()[indx] = it->length;
    //     indx++;
    //   }
    //
    //   struct Error err2 = kernel::NumpyArray_contiguous_copy_from_many_64(
    //     kernel::lib::cpu,   // DERIVE
    //     reinterpret_cast<uint8_t*>(ptr.get()),
    //     data_ptrs,
    //     data_lens.data(),
    //     bytepos.length(),
    //     strides[0],
    //     bytepos.data());
    //   util::handle_error(err2, classname(), identities.get());
    //
    //   return std::make_shared<NumpyArray>(identities,
    //                                       form_.get()->parameters(),
    //                                       ptr,
    //                                       shape,
    //                                       strides,
    //                                       0,
    //                                       dtype_to_itemsize(form_.get()->dtype()),
    //                                       form_.get()->format(),
    //                                       form_.get()->dtype(),
    //                                       kernel::lib::cpu);
    // }
    // else {
    //   // FIXME: this is to peek at the last data buffer
    //   //
    //   std::vector<ssize_t> last_shape = { (ssize_t)data_.back()->length };
    //
    //   return std::make_shared<NumpyArray>(identities,
    //                                       form_.get()->parameters(),
    //                                       data_.back()->ptr,
    //                                       last_shape,
    //                                       strides,
    //                                       0,
    //                                       dtype_to_itemsize(form_.get()->dtype()),
    //                                       form_.get()->format(),
    //                                       form_.get()->dtype(),
    //                                       kernel::lib::cpu);
    //
    // }
  }

  const FormPtr
  NumpyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  // void
  // NumpyArrayBuilder::integer(int64_t x) {
  //   if (!data_.empty()) {
  //     reinterpret_cast<int64_t*>(data_.back()->ptr.get())[length_] = x;
  //     data_.back().get()->length += 1;
  //     length_ += 1;
  //   }
  //   else {
  //     throw std::invalid_argument(
  //       std::string("FormBuilder ") + classname()
  //       + std::string(" needs a data buffer")
  //       + FILENAME(__LINE__));
  //   }
  // }
  //
  // void
  // NumpyArrayBuilder::boolean(bool x) {
  //   if (!data_.empty()) {
  //     reinterpret_cast<bool*>(data_.back()->ptr.get())[length_] = x;
  //     data_.back().get()->length += 1;
  //     length_ += 1;
  //   }
  //   else {
  //     throw std::invalid_argument(
  //       std::string("FormBuilder ") + classname()
  //       + std::string(" needs a data buffer")
  //       + FILENAME(__LINE__));
  //   }
  // }
  //
  // void
  // NumpyArrayBuilder::real(double x) {
  //   if (!data_.empty()) {
  //     reinterpret_cast<double*>(data_.back()->ptr.get())[length_] = x;
  //     data_.back().get()->length += 1;
  //     length_ += 1;
  //   }
  //   else {
  //     throw std::invalid_argument(
  //       std::string("FormBuilder ") + classname()
  //       + std::string(" needs a data buffer")
  //       + FILENAME(__LINE__));
  //   }
  // }

  ///
  RecordArrayBuilder::RecordArrayBuilder(const RecordFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) {
    for (auto const& content : form.get()->contents()) {
      contents_.push_back(formBuilderFromA(content));
    }
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

  // const FormBuilderPtr&
  // RecordArrayBuilder::field_check(const char* key) const {
  //   auto const& pos = distance(keys_.begin(), find(keys_.begin(), keys_.end(), key));
  //   // Note, 'pos' is the number of increments needed to go from first to last.
  //   // The value may be negative if random-access iterators are used and
  //   // first is reachable from last.
  //   if (pos >= (int64_t)keys_.size()) {
  //     // key is not found
  //     throw std::invalid_argument(
  //       std::string("FormBuilder ") + classname()
  //       + std::string(" does not have a content with a key \"")
  //       + std::string(key) + std::string("\"")
  //       + FILENAME(__LINE__));
  //   }
  //   else {
  //     return *std::next(contents_.begin(), pos);
  //   }
  // }

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

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

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

  ///
  UnmaskedArrayBuilder::UnmaskedArrayBuilder(const UnmaskedFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()) { }

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

}
