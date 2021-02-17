// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/builder/TypedArrayBuilder.h"
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
  formBuilderFromA(const FormPtr& form, const DataPtr& data, int64_t length) {
    if (auto downcasted_form = std::dynamic_pointer_cast<BitMaskedForm>(form)) {
      return std::make_shared<BitMaskedArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<ByteMaskedForm>(form)) {
      return std::make_shared<ByteMaskedArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<EmptyForm>(form)) {
      return std::make_shared<EmptyArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<IndexedForm>(form)) {
      return std::make_shared<IndexedArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<IndexedOptionForm>(form)) {
      return std::make_shared<IndexedOptionArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<ListForm>(form)) {
      return std::make_shared<ListArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<ListOffsetForm>(form)) {
      return std::make_shared<ListOffsetArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<NumpyForm>(form)) {
      return std::make_shared<NumpyArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<RecordForm>(form)) {
      return std::make_shared<RecordArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<RegularForm>(form)) {
      return std::make_shared<RegularArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<UnionForm>(form)) {
      return std::make_shared<UnionArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<UnmaskedForm>(form)) {
      return std::make_shared<UnmaskedArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else if (auto downcasted_form = std::dynamic_pointer_cast<VirtualForm>(form)) {
      return std::make_shared<VirtualArrayBuilder>(
        downcasted_form,
        data,
        length);
    }
    else {
      throw std::invalid_argument(
        std::string("Form is not recognised")
        + FILENAME(__LINE__));
    }
  }

  TypedArrayBuilder::TypedArrayBuilder(int64_t initial,
                                       const DataPtr& data,
                                       const int64_t length)
    : initial_(initial),
      builder_(nullptr),
      current_builder_(nullptr),
      data_(data),
      length_(length) { }

  void
  TypedArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (builder_ != nullptr) {
      int64_t offset = builder_.get()->apply(form, data, length);
      data_ = data;
      length_ += offset;

      /// FIXME: remember to put a boundary check here
      assert(length_ < initial_);
    }
    else {
      builder_ = formBuilderFromA(form, data, length);
      data_ = data;
      length_ = length;
    }
  }

  const ContentPtr
  TypedArrayBuilder::snapshot() const {
    return builder_.get()->snapshot();
  }

  void
  TypedArrayBuilder::set_input_buffer(const DataPtr& data) {
    if (builder_ != nullptr) {
      builder_.get()->set_input_buffer(data);
    }
    data_ = data;
  }

  void
  TypedArrayBuilder::set_data_length(const int64_t length) {
    if (builder_ != nullptr) {
      builder_.get()->set_data_length(length);
    }
    length_ = length;
  }

  void
  TypedArrayBuilder::field_check(const char* key) {
    if (builder_ != nullptr) {
      current_builder_ = builder_.get()->field_check(key);
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
  }

  void
  TypedArrayBuilder::integer(int64_t x) {
    if (builder_ != nullptr) {
      if (builder_.get()->accepts(util::dtype::int64)) {
        builder_.get()->integer(x);
      }
      else if (current_builder_ != nullptr  &&
               current_builder_.get()->accepts(util::dtype::int64)) {
        current_builder_.get()->integer(x);
      }
      else {
        throw std::invalid_argument(
          std::string("FormBuilder ") + builder_.get()->classname()
          + std::string(" does not accept integers")
          + FILENAME(__LINE__));
      }
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
  }

  void
  TypedArrayBuilder::boolean(bool x) {
    if (builder_ != nullptr) {
      if (builder_.get()->accepts(util::dtype::boolean)) {
        builder_.get()->boolean(x);
      }
      else if (current_builder_ != nullptr  &&
               current_builder_.get()->accepts(util::dtype::boolean)) {
        current_builder_.get()->boolean(x);
      }
      else {
        throw std::invalid_argument(
          std::string("FormBuilder ") + builder_.get()->classname()
          + std::string(" does not accept boolean")
          + FILENAME(__LINE__));
      }
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
  }

  void
  TypedArrayBuilder::real(double x) {
    if (builder_ != nullptr) {
      if (builder_.get()->accepts(util::dtype::float64)) {
        builder_.get()->real(x);
      }
      else if (current_builder_ != nullptr  &&
               current_builder_.get()->accepts(util::dtype::float64)) {
        current_builder_.get()->real(x);
      }
      else {
        throw std::invalid_argument(
          std::string("FormBuilder ") + builder_.get()->classname()
          + std::string(" does not accept real")
          + FILENAME(__LINE__));
      }
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder is not defined")
        + FILENAME(__LINE__));
    }
  }

  /// @brief
  BitMaskedArrayBuilder::BitMaskedArrayBuilder(const BitMaskedFormPtr& form,
                                               const DataPtr& data,
                                               const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr) { }

  const std::string
  BitMaskedArrayBuilder::classname() const {
    return "BitMaskedArrayBuilder";
  }

  const ContentPtr
  BitMaskedArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      const IndexU8 mask(std::static_pointer_cast<uint8_t>(data_), 0, length_, kernel::lib::cpu);
      return std::make_shared<BitMaskedArray>(Identities::none(),
                                              form_.get()->parameters(),
                                              mask,
                                              content_.get()->snapshot(),
                                              form_.get()->valid_when(),
                                              length_,
                                              form_.get()->lsb_order());
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  BitMaskedArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      // uint8_t itemsize is 1
      return length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form,
                                                 const DataPtr& data,
                                                 const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr) { }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      const Index8 mask(std::static_pointer_cast<int8_t>(data_), 0, length_, kernel::lib::cpu);
      return std::make_shared<ByteMaskedArray>(Identities::none(),
                                               form_.get()->parameters(),
                                               mask,
                                               content_.get()->snapshot(),
                                               form_.get()->valid_when());
     }
     else {
       throw std::invalid_argument(
         std::string("Form of a ") + classname()
         + std::string(" needs another Form as its content")
         + FILENAME(__LINE__));
     }
  }

  int64_t
  ByteMaskedArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      return length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  EmptyArrayBuilder::EmptyArrayBuilder(const EmptyFormPtr& form,
                                       const DataPtr& data,
                                       const int64_t length)
    : form_(form),
      data_(data),
      length_(length) { }

  const std::string
  EmptyArrayBuilder::classname() const {
    return "EmptyArrayBuilder";
  }

  const ContentPtr
  EmptyArrayBuilder::snapshot() const {
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  int64_t
  EmptyArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    throw std::invalid_argument(
      std::string("Form of a ") + classname()
      + std::string(" can not embed another Form")
      + FILENAME(__LINE__));
  }

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const IndexedFormPtr& form,
                                           const DataPtr& data,
                                           const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr) { }

  const std::string
  IndexedArrayBuilder::classname() const {
    return "IndexedArrayBuilder";
  }

  const ContentPtr
  IndexedArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      Index64 index(std::static_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
      return std::make_shared<IndexedArray64>(Identities::none(),
                                              form_.get()->parameters(),
                                              index,
                                              content_.get()->snapshot());
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  IndexedArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      // FIXME: get it from a Form: int64_t itemsize == 8
      return 8 * length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  IndexedOptionArrayBuilder::IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form,
                                                       const DataPtr& data,
                                                       const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr) { }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const ContentPtr
  IndexedOptionArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      Index64 index(std::static_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
      return std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                    form_.get()->parameters(),
                                                    index,
                                                    content_.get()->snapshot());
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  IndexedOptionArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      return 8 * length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  ListArrayBuilder::ListArrayBuilder(const ListFormPtr& form,
                                     const DataPtr& data,
                                     int64_t length,
                                     bool copyarrays)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr),
      copyarrays_(copyarrays) { }

  const std::string
  ListArrayBuilder::classname() const {
    return "ListArrayBuilder";
  }

  const ContentPtr
  ListArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      Index64 starts(std::static_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
      Index64 stops(std::static_pointer_cast<int64_t>(data_), length_, length_, kernel::lib::cpu);
      return std::make_shared<ListArray64>(Identities::none(),
                                           form_.get()->parameters(),
                                           starts,
                                           stops,
                                           content_.get()->snapshot());
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  ListArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      return 8 * length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  ListOffsetArrayBuilder::ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                                                 const DataPtr& data,
                                                 int64_t length,
                                                 bool copyarrays)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr),
      copyarrays_(copyarrays) { }

  const std::string
  ListOffsetArrayBuilder::classname() const {
    return "ListOffsetArrayBuilder";
  }

  const ContentPtr
  ListOffsetArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      Index64 offsets(std::static_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
      return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                 form_.get()->parameters(),
                                                 offsets,
                                                 content_.get()->snapshot());
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  ListOffsetArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
      return 8 * length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
  }

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const NumpyFormPtr& form,
                                       const DataPtr& data,
                                       const int64_t length,
                                       bool copyarrays)
    : form_(form),
      copyarrays_(copyarrays),
      length_(length) {
      if(data != nullptr) {
        data_.push_back(std::unique_ptr<Data>(new Data(data, length)));
      }
    }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const ContentPtr
  NumpyArrayBuilder::snapshot() const {
    auto identities = Identities::none();

    // FIXME: check that length_ is always equal to
    // Sum s = std::for_each(data_.begin(), data_.end(), Sum());

    // FIXME:
    // nothing to be done if all data pointers hold the same memory address
    // otherwise, copy the data into a single continous memory
    //
    std::vector<ssize_t> shape = { (ssize_t)length_ };
    std::vector<ssize_t> strides = { (ssize_t)dtype_to_itemsize(form_.get()->dtype()) };

    if (copyarrays_) {
      Index64 bytepos(shape[0]);
      struct Error err1 = kernel::NumpyArray_contiguous_init_64(
        kernel::lib::cpu,   // DERIVE
        bytepos.data(),
        shape[0],
        strides[0]);
      util::handle_error(err1, classname(), identities.get());

      std::shared_ptr<void> ptr(
        kernel::malloc<void>(kernel::lib::cpu, bytepos.length()*strides[0]));

      struct Error err2 = kernel::NumpyArray_contiguous_copy_64(
        kernel::lib::cpu,   // DERIVE
        reinterpret_cast<uint8_t*>(ptr.get()),
        reinterpret_cast<uint8_t*>(data_.front()->ptr.get()),
        bytepos.length(),
        strides[0],
        bytepos.data());
      util::handle_error(err2, classname(), identities.get());

      return std::make_shared<NumpyArray>(identities,
                                          form_.get()->parameters(),
                                          ptr,
                                          shape,
                                          strides,
                                          0,
                                          dtype_to_itemsize(form_.get()->dtype()),
                                          form_.get()->format(),
                                          form_.get()->dtype(),
                                          kernel::lib::cpu);
    }
    else {
      return std::make_shared<NumpyArray>(identities,
                                          form_.get()->parameters(),
                                          data_.front()->ptr,
                                          shape,
                                          strides,
                                          0,
                                          dtype_to_itemsize(form_.get()->dtype()),
                                          form_.get()->format(),
                                          form_.get()->dtype(),
                                          kernel::lib::cpu);

    }
  }

  int64_t
  NumpyArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->equal(form, true, true, true, true)) {
      data_.push_back(std::unique_ptr<Data>(new Data(data, length)));
      length_ += length;
      return dtype_to_itemsize(form_.get()->dtype()) * length_;
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" can not embed another Form")
        + FILENAME(__LINE__));
    }
  }

  bool
  NumpyArrayBuilder::accepts(util::dtype dt) const {
    return (form_.get()->dtype() == dt);
  }

  int64_t
  NumpyArrayBuilder::length() const {
    return data_.back().get()->length;
  }

  void
  NumpyArrayBuilder::integer(int64_t x) {
    if (!data_.empty()) {
      reinterpret_cast<int64_t*>(data_.back()->ptr.get())[length_] = x;
      data_.back().get()->length += 1;
      length_ += 1;
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder ") + classname()
        + std::string(" needs a data buffer")
        + FILENAME(__LINE__));
    }
  }

  void
  NumpyArrayBuilder::boolean(bool x) {
    if (!data_.empty()) {
      reinterpret_cast<bool*>(data_.back()->ptr.get())[length_] = x;
      data_.back().get()->length += 1;
      length_ += 1;
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder ") + classname()
        + std::string(" needs a data buffer")
        + FILENAME(__LINE__));
    }
  }

  void
  NumpyArrayBuilder::real(double x) {
    if (!data_.empty()) {
      reinterpret_cast<double*>(data_.back()->ptr.get())[length_] = x;
      data_.back().get()->length += 1;
      length_ += 1;
    }
    else {
      throw std::invalid_argument(
        std::string("FormBuilder ") + classname()
        + std::string(" needs a data buffer")
        + FILENAME(__LINE__));
    }
  }

  ///
  RecordArrayBuilder::RecordArrayBuilder(const RecordFormPtr& form,
                                         const DataPtr& data,
                                         const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      contents_(std::vector<FormBuilderPtr>()),
      keys_(std::vector<std::string>()) { }

  const std::string
  RecordArrayBuilder::classname() const {
    return "RecordArrayBuilder";
  }

  const ContentPtr
  RecordArrayBuilder::snapshot() const {
    if (! contents_.empty()) {
      ContentPtrVec contents;
      util::RecordLookupPtr recordlookup =
        std::make_shared<util::RecordLookup>();
      for (size_t i = 0;  i < contents_.size();  i++) {
        contents.push_back(contents_[i].get()->snapshot());
        recordlookup.get()->push_back(keys_[i]);
      }
      std::vector<ArrayCachePtr> caches;  // nothing is virtual here
      return std::make_shared<RecordArray>(Identities::none(),
                                           form_.get()->parameters(),
                                           contents,
                                           recordlookup,
                                           length_,
                                           caches);
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  RecordArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (keys_.empty()) {
      keys_.push_back(*form.get()->form_key());
      contents_.push_back(formBuilderFromA(form, data, length));
    }
    else {
      const std::string& key = *form.get()->form_key();
      auto pos = distance(keys_.begin(), find(keys_.begin(), keys_.end(), key));
      if (pos >= keys_.size()) {
        // key is not found
        keys_.push_back(*form.get()->form_key());
        contents_.push_back(formBuilderFromA(form, data, length));
      }
      else {
        return contents_[pos].get()->apply(form, data, length);
      }
    }
    /// FIXME: every record's data buffer has a different length
    return length_;
  }

  const FormBuilderPtr&
  RecordArrayBuilder::field_check(const char* key) const {
    auto pos = distance(keys_.begin(), find(keys_.begin(), keys_.end(), key));
    if (pos >= keys_.size()) {
      // key is not found
      throw std::invalid_argument(
        std::string("FormBuilder ") + classname()
        + std::string(" does not have a content with a key \"")
        + std::string(key) + std::string("\"")
        + FILENAME(__LINE__));
    }
    else {
      return contents_[pos];
    }
  }

  ///
  RegularArrayBuilder::RegularArrayBuilder(const RegularFormPtr& form,
                                           const DataPtr& data,
                                           const int64_t length)
    : form_(form),
      data_(data),
      length_(length),
      content_(nullptr) { }

  const std::string
  RegularArrayBuilder::classname() const {
    return "RegularArrayBuilder";
  }

  const ContentPtr
  RegularArrayBuilder::snapshot() const {
    if(content_ != nullptr) {
      return std::make_shared<RegularArray>(Identities::none(),
                                            form_.get()->parameters(),
                                            content_.get()->snapshot(),
                                            length_);
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" needs another Form as its content")
        + FILENAME(__LINE__));
    }
  }

  int64_t
  RegularArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    if (form_.get()->content().get()->equal(form, false, false, false, false)) {
      content_ = formBuilderFromA(form, data, length);
    }
    else {
      throw std::invalid_argument(
        std::string("Form of a ") + classname()
        + std::string(" expects another Form")
        + FILENAME(__LINE__));
    }
    /// FIXME:
    return length_;
  }

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form,
                                       const DataPtr& data,
                                       const int64_t length)
    : form_(form),
      data_(data),
      length_(length) { }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const ContentPtr
  UnionArrayBuilder::snapshot() const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
      // return std::make_shared<UnionArray8_64>(Identities::none(),
      //                                         form_.get()->parameters());
  }

  int64_t
  UnionArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    throw std::invalid_argument(
      std::string("Form of a ") + classname()
      + std::string(" can not embed another Form")
      + FILENAME(__LINE__));
  }

  ///
  UnmaskedArrayBuilder::UnmaskedArrayBuilder(const UnmaskedFormPtr& form,
                                             const DataPtr& data,
                                             const int64_t length)
    : form_(form),
      data_(data),
      length_(length) { }

  const std::string
  UnmaskedArrayBuilder::classname() const {
    return "UnmaskedArrayBuilder";
  }

  const ContentPtr
  UnmaskedArrayBuilder::snapshot() const {
    // FIXME
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  int64_t
  UnmaskedArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    throw std::invalid_argument(
      std::string("Form of a ") + classname()
      + std::string(" can not embed another Form")
      + FILENAME(__LINE__));
  }

  ///
  VirtualArrayBuilder::VirtualArrayBuilder(const VirtualFormPtr& form,
                                           const DataPtr& data,
                                           const int64_t length)
    : form_(form),
      data_(data),
      length_(length) { }

  const std::string
  VirtualArrayBuilder::classname() const {
    return "VirtualArrayBuilder";
  }

  const ContentPtr
  VirtualArrayBuilder::snapshot() const {
    // FIXME:
    return std::make_shared<EmptyArray>(Identities::none(),
                                        form_.get()->parameters());
  }

  int64_t
  VirtualArrayBuilder::apply(const FormPtr& form, const DataPtr& data, const int64_t length) {
    throw std::invalid_argument(
      std::string("Form of a ") + classname()
      + std::string(" can not embed another Form")
      + FILENAME(__LINE__));
  }

}
