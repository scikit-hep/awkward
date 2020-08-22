// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/BitMaskedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/BitMaskedArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/None.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"

#include "awkward/array/BitMaskedArray.h"

namespace awkward {
  ////////// BitMaskedForm

  BitMaskedForm::BitMaskedForm(bool has_identities,
                               const util::Parameters& parameters,
                               const FormKey& form_key,
                               Index::Form mask,
                               const FormPtr& content,
                               bool valid_when,
                               bool lsb_order)
      : Form(has_identities, parameters, form_key)
      , mask_(mask)
      , content_(content)
      , valid_when_(valid_when)
      , lsb_order_(lsb_order) { }

  Index::Form
  BitMaskedForm::mask() const {
    return mask_;
  }

  const FormPtr
  BitMaskedForm::content() const {
    return content_;
  }

  bool
  BitMaskedForm::valid_when() const {
    return valid_when_;
  }

  bool
  BitMaskedForm::lsb_order() const {
    return lsb_order_;
  }

  const TypePtr
  BitMaskedForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<OptionType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               content_.get()->type(typestrs));
  }

  void
  BitMaskedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("BitMaskedArray");
    builder.field("mask");
    builder.string(Index::form2str(mask_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    builder.field("valid_when");
    builder.boolean(valid_when_);
    builder.field("lsb_order");
    builder.boolean(lsb_order_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  BitMaskedForm::shallow_copy() const {
    return std::make_shared<BitMaskedForm>(has_identities_,
                                           parameters_,
                                           form_key_,
                                           mask_,
                                           content_,
                                           valid_when_,
                                           lsb_order_);
  }

  const std::string
  BitMaskedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  BitMaskedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  BitMaskedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  const std::pair<int64_t, int64_t>
  BitMaskedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  BitMaskedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  BitMaskedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  BitMaskedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  BitMaskedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  BitMaskedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  BitMaskedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  BitMaskedForm::equal(const FormPtr& other,
                       bool check_identities,
                       bool check_parameters,
                       bool check_form_key,
                       bool compatibility_check) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (check_form_key  &&
        !form_key_equals(other.get()->form_key())) {
      return false;
    }
    if (BitMaskedForm* t = dynamic_cast<BitMaskedForm*>(other.get())) {
      return (mask_ == t->mask()  &&
              content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check)  &&
              valid_when_ == t->valid_when()  &&
              lsb_order_ == t->lsb_order());
    }
    else {
      return false;
    }
  }

  const FormPtr
  BitMaskedForm::getitem_field(const std::string& key) const {
    return content_.get()->getitem_field(key);
  }

  ////////// BitMaskedArray

  BitMaskedArray::BitMaskedArray(const IdentitiesPtr& identities,
                                 const util::Parameters& parameters,
                                 const IndexU8& mask,
                                 const ContentPtr& content,
                                 bool valid_when,
                                 int64_t length,
                                 bool lsb_order)
      : Content(identities, parameters)
      , mask_(mask)
      , content_(content)
      , valid_when_(valid_when != 0)
      , length_(length)
      , lsb_order_(lsb_order) {
    int64_t bitlength = ((length / 8) + ((length % 8) != 0));
    if (mask.length() < bitlength) {
      throw std::invalid_argument(
        std::string("BitMaskedArray mask must not be shorter than its ceil(length / 8.0)")
        + FILENAME(__LINE__));
    }
    if (content.get()->length() < length) {
      throw std::invalid_argument(
        std::string("BitMaskedArray content must not be shorter than its length")
        + FILENAME(__LINE__));
    }
  }

  const IndexU8
  BitMaskedArray::mask() const {
    return mask_;
  }

  const ContentPtr
  BitMaskedArray::content() const {
    return content_;
  }

  bool
  BitMaskedArray::valid_when() const {
    return valid_when_;
  }

  bool
  BitMaskedArray::lsb_order() const {
    return lsb_order_;
  }

  const ContentPtr
  BitMaskedArray::project() const {
    return toByteMaskedArray().get()->project();
  }

  const ContentPtr
  BitMaskedArray::project(const Index8& mask) const {
    return toByteMaskedArray().get()->project(mask);
  }

  const Index8
  BitMaskedArray::bytemask() const {
    Index8 bytemask(mask_.length() * 8);
    struct Error err = kernel::BitMaskedArray_to_ByteMaskedArray(
      kernel::lib::cpu,   // DERIVE
      bytemask.data(),
      mask_.data(),
      mask_.length(),
      valid_when_,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return bytemask.getitem_range_nowrap(0, length_);
  }

  const ContentPtr
  BitMaskedArray::simplify_optiontype() const {
    if (dynamic_cast<IndexedArray32*>(content_.get())        ||
        dynamic_cast<IndexedArrayU32*>(content_.get())       ||
        dynamic_cast<IndexedArray64*>(content_.get())        ||
        dynamic_cast<IndexedOptionArray32*>(content_.get())  ||
        dynamic_cast<IndexedOptionArray64*>(content_.get())  ||
        dynamic_cast<ByteMaskedArray*>(content_.get())       ||
        dynamic_cast<BitMaskedArray*>(content_.get())        ||
        dynamic_cast<UnmaskedArray*>(content_.get())) {
      ContentPtr step1 = toIndexedOptionArray64();
      IndexedOptionArray64* step2 =
        dynamic_cast<IndexedOptionArray64*>(step1.get());
      return step2->simplify_optiontype();
    }
    else {
      return shallow_copy();
    }
  }

  const std::shared_ptr<ByteMaskedArray>
  BitMaskedArray::toByteMaskedArray() const {
    Index8 bytemask(mask_.length() * 8);
    struct Error err = kernel::BitMaskedArray_to_ByteMaskedArray(
      kernel::lib::cpu,   // DERIVE
      bytemask.data(),
      mask_.data(),
      mask_.length(),
      false,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<ByteMaskedArray>(
      identities_,
      parameters_,
      bytemask.getitem_range_nowrap(0, length_),
      content_,
      valid_when_);
  }

  const std::shared_ptr<IndexedOptionArray64>
  BitMaskedArray::toIndexedOptionArray64() const {
    Index64 index(mask_.length() * 8);
    struct Error err = kernel::BitMaskedArray_to_IndexedOptionArray64(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      mask_.data(),
      mask_.length(),
      valid_when_,
      lsb_order_);
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<IndexedOptionArray64>(
      identities_,
      parameters_,
      index.getitem_range_nowrap(0, length_),
      content_);
  }

  const std::string
  BitMaskedArray::classname() const {
    return "BitMaskedArray";
  }

  void
  BitMaskedArray::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(
          failure("content and its identities must have the same length",
                  kSliceNone,
                  kSliceNone,
                  FILENAME_C(__LINE__)),
          classname(),
          identities_.get());
      }
      if (Identities32* rawidentities =
          dynamic_cast<Identities32*>(identities.get())) {
        std::shared_ptr<Identities32> subidentities =
          std::make_shared<Identities32>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width(),
                                         content_.get()->length());
        Identities32* rawsubidentities =
          reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = kernel::Identities_extend<int32_t>(
          kernel::lib::cpu,   // DERIVE
          rawsubidentities->data(),
          rawidentities->data(),
          rawidentities->length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else if (Identities64* rawidentities =
               dynamic_cast<Identities64*>(identities.get())) {
        std::shared_ptr<Identities64> subidentities =
          std::make_shared<Identities64>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width(),
                                         content_.get()->length());
        Identities64* rawsubidentities =
          reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = kernel::Identities_extend<int64_t>(
          kernel::lib::cpu,   // DERIVE
          rawsubidentities->data(),
          rawidentities->data(),
          rawidentities->length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized Identities specialization")
          + FILENAME(__LINE__));
      }
    }
    identities_ = identities;
  }

  void
  BitMaskedArray::setidentities() {
    if (length() <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_Identities<int32_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      IdentitiesPtr newidentities =
        std::make_shared<Identities64>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities64* rawidentities =
        reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = kernel::new_Identities<int64_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  const TypePtr
  BitMaskedArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  BitMaskedArray::form(bool materialize) const {
    return std::make_shared<BitMaskedForm>(identities_.get() != nullptr,
                                           parameters_,
                                           FormKey(nullptr),
                                           mask_.form(),
                                           content_.get()->form(materialize),
                                           valid_when_,
                                           lsb_order_);
  }

  bool
  BitMaskedArray::has_virtual_form() const {
    return content_.get()->has_virtual_form();
  }

  bool
  BitMaskedArray::has_virtual_length() const {
    return content_.get()->has_virtual_length();
  }

  const std::string
  BitMaskedArray::tostring_part(const std::string& indent,
                                const std::string& pre,
                                const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " valid_when=\""
        << (valid_when_ ? "true" : "false") << "\" length=\"" << length_
        << "\" lsb_order=\"" << (lsb_order_ ? "true" : "false") << "\">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << mask_.tostring_part(
             indent + std::string("    "), "<mask>", "</mask>\n");
    out << content_.get()->tostring_part(
             indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  BitMaskedArray::tojson_part(ToJson& builder,
                              bool include_beginendlist) const {
    int64_t len = length();
    check_for_iteration();
    if (include_beginendlist) {
      builder.beginlist();
    }
    for (int64_t i = 0;  i < len;  i++) {
      getitem_at_nowrap(i).get()->tojson_part(builder, true);
    }
    if (include_beginendlist) {
      builder.endlist();
    }
  }

  void
  BitMaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    mask_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t
  BitMaskedArray::length() const {
    return length_;
  }

  const ContentPtr
  BitMaskedArray::shallow_copy() const {
    return std::make_shared<BitMaskedArray>(identities_,
                                            parameters_,
                                            mask_,
                                            content_,
                                            valid_when_,
                                            length_,
                                            lsb_order_);
  }

  const ContentPtr
  BitMaskedArray::deep_copy(bool copyarrays,
                            bool copyindexes,
                            bool copyidentities) const {
    IndexU8 mask = copyindexes ? mask_.deep_copy() : mask_;
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<BitMaskedArray>(identities,
                                            parameters_,
                                            mask,
                                            content,
                                            valid_when_,
                                            length_,
                                            lsb_order_);
  }

  void
  BitMaskedArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < length()) {
      util::handle_error(
        failure("len(identities) < len(array)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
  }

  const ContentPtr
  BitMaskedArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const ContentPtr
  BitMaskedArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  BitMaskedArray::getitem_at_nowrap(int64_t at) const {
    int64_t bitat = at / 8;
    int64_t shift = at % 8;
    uint8_t byte = mask_.getitem_at_nowrap(bitat);
    uint8_t asbool = (lsb_order_
                          ? ((byte >> ((uint8_t)shift)) & ((uint8_t)1))
                          : ((byte << ((uint8_t)shift)) & ((uint8_t)128)));
    if ((asbool != 0) == valid_when_) {
      return content_.get()->getitem_at_nowrap(at);
    }
    else {
      return none;
    }
  }

  const ContentPtr
  BitMaskedArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range", kSliceNone, stop, FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const ContentPtr
  BitMaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    int64_t bitstart = start / 8;
    int64_t remainder = start % 8;
    if (remainder == 0) {
      IdentitiesPtr identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_range_nowrap(start, stop);
      }
      int64_t length = stop - start;
      int64_t bitlength = length / 8;
      int64_t remainder = length % 8;
      int64_t bitstop = bitstart + (bitlength + (remainder != 0));
      return std::make_shared<BitMaskedArray>(
        identities,
        parameters_,
        mask_.getitem_range_nowrap(bitstart, bitstop),
        content_.get()->getitem_range_nowrap(start, stop),
        valid_when_,
        length,
        lsb_order_);
    }
    else {
      return toByteMaskedArray().get()->getitem_range_nowrap(start, stop);
    }
  }

  const ContentPtr
  BitMaskedArray::getitem_field(const std::string& key) const {
    return std::make_shared<BitMaskedArray>(
      identities_,
      util::Parameters(),
      mask_,
      content_.get()->getitem_field(key),
      valid_when_,
      length_,
      lsb_order_);
  }

  const ContentPtr
  BitMaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<BitMaskedArray>(
      identities_,
      util::Parameters(),
      mask_,
      content_.get()->getitem_fields(keys),
      valid_when_,
      length_,
      lsb_order_);
  }

  const ContentPtr
  BitMaskedArray::getitem_next(const SliceItemPtr& head,
                               const Slice& tail,
                               const Index64& advanced) const {
    return toByteMaskedArray().get()->getitem_next(head, tail, advanced);
  }

  const ContentPtr
  BitMaskedArray::carry(const Index64& carry, bool allow_lazy) const {
    return toByteMaskedArray().get()->carry(carry, allow_lazy);
  }

  int64_t
  BitMaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  BitMaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  BitMaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  BitMaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  BitMaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string
  BitMaskedArray::validityerror(const std::string& path) const {
    if (mask_.length() * 8 < length_) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string("len(mask) * 8 < length")
              + FILENAME(__LINE__));
    }
    else if (content_.get()->length() < length_) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string("len(content) < length")
              + FILENAME(__LINE__));
    }
    else {
      return content_.get()->validityerror(path + std::string(".content"));
    }
  }

  const ContentPtr
  BitMaskedArray::shallow_simplify() const {
    return simplify_optiontype();
  }

  const ContentPtr
  BitMaskedArray::num(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->num(axis, depth);
  }

  const std::pair<Index64, ContentPtr>
  BitMaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->offsets_and_flattened(axis, depth);
  }

  bool
  BitMaskedArray::mergeable(const ContentPtr& other, bool mergebool) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return mergeable(raw->array(), mergebool);
    }

    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }

    if (dynamic_cast<EmptyArray*>(other.get())  ||
        dynamic_cast<UnionArray8_32*>(other.get())  ||
        dynamic_cast<UnionArray8_U32*>(other.get())  ||
        dynamic_cast<UnionArray8_64*>(other.get())) {
      return true;
    }

    if (IndexedArray32* rawother =
        dynamic_cast<IndexedArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArrayU32* rawother =
             dynamic_cast<IndexedArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArray64* rawother =
             dynamic_cast<IndexedArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray32* rawother =
             dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray64* rawother =
             dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ByteMaskedArray* rawother =
             dynamic_cast<ByteMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (BitMaskedArray* rawother =
             dynamic_cast<BitMaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (UnmaskedArray* rawother =
             dynamic_cast<UnmaskedArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return content_.get()->mergeable(other, mergebool);
    }
  }

  const ContentPtr
  BitMaskedArray::reverse_merge(const ContentPtr& other) const {
    ContentPtr indexedoptionarray = toIndexedOptionArray64();
    IndexedOptionArray64* raw =
      dynamic_cast<IndexedOptionArray64*>(indexedoptionarray.get());
    return raw->reverse_merge(other);
  }

  const ContentPtr
  BitMaskedArray::merge(const ContentPtr& other) const {
    return toIndexedOptionArray64().get()->merge(other);
  }

  const SliceItemPtr
  BitMaskedArray::asslice() const {
    return toIndexedOptionArray64().get()->asslice();
  }

  const ContentPtr
  BitMaskedArray::fillna(const ContentPtr& value) const {
    return toIndexedOptionArray64().get()->fillna(value);
  }

  const ContentPtr
  BitMaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->rpad(target, axis, depth);
  }

  const ContentPtr
  BitMaskedArray::rpad_and_clip(int64_t target,
                                int64_t axis,
                                int64_t depth) const {
    return toByteMaskedArray().get()->rpad_and_clip(target, axis, depth);
  }

  const ContentPtr
  BitMaskedArray::reduce_next(const Reducer& reducer,
                              int64_t negaxis,
                              const Index64& starts,
                              const Index64& shifts,
                              const Index64& parents,
                              int64_t outlength,
                              bool mask,
                              bool keepdims) const {
    return toByteMaskedArray().get()->reduce_next(reducer,
                                                  negaxis,
                                                  starts,
                                                  shifts,
                                                  parents,
                                                  outlength,
                                                  mask,
                                                  keepdims);
  }

  const ContentPtr
  BitMaskedArray::localindex(int64_t axis, int64_t depth) const {
    return toByteMaskedArray().get()->localindex(axis, depth);
  }

  const ContentPtr
  BitMaskedArray::combinations(int64_t n,
                               bool replacement,
                               const util::RecordLookupPtr& recordlookup,
                               const util::Parameters& parameters,
                               int64_t axis,
                               int64_t depth) const {
    return toByteMaskedArray().get()->combinations(n,
                                                   replacement,
                                                   recordlookup,
                                                   parameters,
                                                   axis,
                                                   depth);
  }

  const ContentPtr
  BitMaskedArray::sort_next(int64_t negaxis,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength,
                            bool ascending,
                            bool stable,
                            bool keepdims) const {
    return toByteMaskedArray().get()->sort_next(negaxis,
                                                starts,
                                                parents,
                                                outlength,
                                                ascending,
                                                stable,
                                                keepdims);
  }

  const ContentPtr
  BitMaskedArray::argsort_next(int64_t negaxis,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength,
                               bool ascending,
                               bool stable,
                               bool keepdims) const {
    return toByteMaskedArray().get()->argsort_next(negaxis,
                                                   starts,
                                                   parents,
                                                   outlength,
                                                   ascending,
                                                   stable,
                                                   keepdims);
  }

  const ContentPtr
  BitMaskedArray::getitem_next(const SliceAt& at,
                               const Slice& tail,
                               const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: BitMaskedArraygetitem_next(at)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  BitMaskedArray::getitem_next(const SliceRange& range,
                               const Slice& tail,
                               const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: BitMaskedArraygetitem_next(range)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  BitMaskedArray::getitem_next(const SliceArray64& array,
                               const Slice& tail,
                               const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: BitMaskedArraygetitem_next(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  BitMaskedArray::getitem_next(const SliceJagged64& jagged,
                               const Slice& tail,
                               const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: BitMaskedArraygetitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  BitMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceArray64& slicecontent,
                                      const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts,
                                                          slicestops,
                                                          slicecontent,
                                                          tail);
  }

  const ContentPtr
  BitMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceMissing64& slicecontent,
                                      const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts,
                                                          slicestops,
                                                          slicecontent,
                                                          tail);
  }

  const ContentPtr
  BitMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceJagged64& slicecontent,
                                      const Slice& tail) const {
    return toByteMaskedArray().get()->getitem_next_jagged(slicestarts,
                                                          slicestops,
                                                          slicecontent,
                                                          tail);
  }

  const ContentPtr
  BitMaskedArray::copy_to(kernel::lib ptr_lib) const {
    IndexU8 mask = mask_.copy_to(ptr_lib);
    ContentPtr content = content_.get()->copy_to(ptr_lib);
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<BitMaskedArray>(identities,
                                            parameters_,
                                            mask,
                                            content,
                                            valid_when_,
                                            length_,
                                            lsb_order_);
  }

  const ContentPtr
  BitMaskedArray::numbers_to_type(const std::string& name) const {
    return toByteMaskedArray().get()->numbers_to_type(name);
  }

}
