// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/UnmaskedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/UnmaskedArray.cpp", line)

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
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/VirtualArray.h"
#include "awkward/array/RegularArray.h"

#include "awkward/array/UnmaskedArray.h"

namespace awkward {
  ////////// UnmaskedForm

  UnmaskedForm::UnmaskedForm(bool has_identities,
                             const util::Parameters& parameters,
                             const FormKey& form_key,
                             const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , content_(content) { }

  const FormPtr
  UnmaskedForm::content() const {
    return content_;
  }

  const TypePtr
  UnmaskedForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<OptionType>(
      parameters_,
      util::gettypestr(parameters_, typestrs),
      content_.get()->type(typestrs));
  }

  void
  UnmaskedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("UnmaskedArray");
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  UnmaskedForm::shallow_copy() const {
    return std::make_shared<UnmaskedForm>(has_identities_,
                                          parameters_,
                                          form_key_,
                                          content_);
  }

  const std::string
  UnmaskedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  UnmaskedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  UnmaskedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  const std::pair<int64_t, int64_t>
  UnmaskedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  UnmaskedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  UnmaskedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  UnmaskedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  UnmaskedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  UnmaskedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  UnmaskedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  UnmaskedForm::equal(const FormPtr& other,
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
    if (UnmaskedForm* t = dynamic_cast<UnmaskedForm*>(other.get())) {
      return (content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check));
    }
    else {
      return false;
    }
  }

  const FormPtr
  UnmaskedForm::getitem_field(const std::string& key) const {
    return content_.get()->getitem_field(key);
  }

  ////////// UnmaskedArray

  UnmaskedArray::UnmaskedArray(const IdentitiesPtr& identities,
                               const util::Parameters& parameters,
                               const ContentPtr& content)
      : Content(identities, parameters)
      , content_(content) { }

  const ContentPtr
  UnmaskedArray::content() const {
    return content_;
  }

  const ContentPtr
  UnmaskedArray::project() const {
    return content_;
  }

  const ContentPtr
  UnmaskedArray::project(const Index8& mask) const {
    return std::make_shared<ByteMaskedArray>(Identities::none(),
                                             util::Parameters(),
                                             mask,
                                             content_,
                                             false).get()->project();
  }

  const Index8
  UnmaskedArray::bytemask() const {
    Index8 out(length());
    struct Error err = kernel::zero_mask8(
      kernel::lib::cpu,   // DERIVE
      out.data(),
      length());
    util::handle_error(err, classname(), identities_.get());
    return out;
  }

  const ContentPtr
  UnmaskedArray::simplify_optiontype() const {
    if (dynamic_cast<IndexedArray32*>(content_.get())        ||
        dynamic_cast<IndexedArrayU32*>(content_.get())       ||
        dynamic_cast<IndexedArray64*>(content_.get())        ||
        dynamic_cast<IndexedOptionArray32*>(content_.get())  ||
        dynamic_cast<IndexedOptionArray64*>(content_.get())  ||
        dynamic_cast<ByteMaskedArray*>(content_.get())       ||
        dynamic_cast<BitMaskedArray*>(content_.get())        ||
        dynamic_cast<UnmaskedArray*>(content_.get())) {
      return content_;
    }
    else {
      return shallow_copy();
    }
  }

  const ContentPtr
  UnmaskedArray::toIndexedOptionArray64() const {
    Index64 index(length());
    struct Error err = kernel::carry_arange<int64_t>(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      length());
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<IndexedOptionArray64>(identities_,
                                                  parameters_,
                                                  index,
                                                  content_);
  }

  const std::string
  UnmaskedArray::classname() const {
    return "UnmaskedArray";
  }

  void
  UnmaskedArray::setidentities(const IdentitiesPtr& identities) {
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
          std::string("unrecognized Identities specialization") + FILENAME(__LINE__));
      }
    }
    identities_ = identities;
  }

  void
  UnmaskedArray::setidentities() {
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
  UnmaskedArray::type(const std::map<std::string,
                      std::string>& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  UnmaskedArray::form(bool materialize) const {
    return std::make_shared<UnmaskedForm>(identities_.get() != nullptr,
                                          parameters_,
                                          FormKey(nullptr),
                                          content_.get()->form(materialize));
  }

  bool
  UnmaskedArray::has_virtual_form() const {
    return content_.get()->has_virtual_form();
  }

  bool
  UnmaskedArray::has_virtual_length() const {
    return content_.get()->has_virtual_length();
  }

  const std::string
  UnmaskedArray::tostring_part(const std::string& indent,
                               const std::string& pre,
                               const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << content_.get()->tostring_part(
             indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  UnmaskedArray::tojson_part(ToJson& builder,
                             bool include_beginendlist) const {
    content_.get()->tojson_part(builder, include_beginendlist);
  }

  void
  UnmaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    content_.get()->nbytes_part(largest);
  }

  int64_t
  UnmaskedArray::length() const {
    return content_.get()->length();
  }

  const ContentPtr
  UnmaskedArray::shallow_copy() const {
    return std::make_shared<UnmaskedArray>(identities_, parameters_, content_);
  }

  const ContentPtr
  UnmaskedArray::deep_copy(bool copyarrays,
                           bool copyindexes,
                           bool copyidentities) const {
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<UnmaskedArray>(identities, parameters_, content);
  }

  void
  UnmaskedArray::check_for_iteration() const {
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
  UnmaskedArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const ContentPtr
  UnmaskedArray::getitem_at(int64_t at) const {
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
  UnmaskedArray::getitem_at_nowrap(int64_t at) const {
    return content_.get()->getitem_at_nowrap(at);
  }

  const ContentPtr
  UnmaskedArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range",
                kSliceNone,
                stop,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const ContentPtr
  UnmaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<UnmaskedArray>(
      identities,
      parameters_,
      content_.get()->getitem_range_nowrap(start, stop));
  }

  const ContentPtr
  UnmaskedArray::getitem_field(const std::string& key) const {
    return std::make_shared<UnmaskedArray>(
      identities_,
      util::Parameters(),
      content_.get()->getitem_field(key));
  }

  const ContentPtr
  UnmaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<UnmaskedArray>(
      identities_,
      util::Parameters(),
      content_.get()->getitem_fields(keys));
  }

  const ContentPtr
  UnmaskedArray::getitem_next(const SliceItemPtr& head,
                              const Slice& tail,
                              const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||
             dynamic_cast<SliceRange*>(head.get())  ||
             dynamic_cast<SliceArray64*>(head.get())  ||
             dynamic_cast<SliceJagged64*>(head.get())) {
      UnmaskedArray out2(identities_,
                         parameters_,
                         content_.get()->getitem_next(head, tail, advanced));
      return out2.simplify_optiontype();
    }
    else if (SliceEllipsis* ellipsis =
             dynamic_cast<SliceEllipsis*>(head.get())) {
      return Content::getitem_next(*ellipsis, tail, advanced);
    }
    else if (SliceNewAxis* newaxis =
             dynamic_cast<SliceNewAxis*>(head.get())) {
      return Content::getitem_next(*newaxis, tail, advanced);
    }
    else if (SliceField* field =
             dynamic_cast<SliceField*>(head.get())) {
      return Content::getitem_next(*field, tail, advanced);
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      return Content::getitem_next(*fields, tail, advanced);
    }
    else if (SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      return Content::getitem_next(*missing, tail, advanced);
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized slice type") + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  UnmaskedArray::carry(const Index64& carry, bool allow_lazy) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<UnmaskedArray>(identities,
                                           parameters_,
                                           content_.get()->carry(carry, allow_lazy));
  }

  int64_t
  UnmaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  UnmaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  UnmaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  UnmaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  UnmaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string
  UnmaskedArray::validityerror(const std::string& path) const {
    return content_.get()->validityerror(path + std::string(".content"));
  }

  const ContentPtr
  UnmaskedArray::shallow_simplify() const {
    return simplify_optiontype();
  }

  const ContentPtr
  UnmaskedArray::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else {
      return std::make_shared<UnmaskedArray>(Identities::none(),
                                             util::Parameters(),
                                             content_.get()->num(posaxis, depth));
    }
  }

  const std::pair<Index64, ContentPtr>
  UnmaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else {
      std::pair<Index64, ContentPtr> offsets_flattened =
        content_.get()->offsets_and_flattened(posaxis, depth);
      Index64 offsets = offsets_flattened.first;
      ContentPtr flattened = offsets_flattened.second;
      if (offsets.length() == 0) {
        return std::pair<Index64, ContentPtr>(
          offsets,
          std::make_shared<UnmaskedArray>(Identities::none(),
                                          util::Parameters(),
                                          flattened));
      }
      else {
        return offsets_flattened;
      }
    }
  }

  bool
  UnmaskedArray::mergeable(const ContentPtr& other, bool mergebool) const {
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
  UnmaskedArray::reverse_merge(const ContentPtr& other) const {
    ContentPtr indexedoptionarray = toIndexedOptionArray64();
    IndexedOptionArray64* raw =
      dynamic_cast<IndexedOptionArray64*>(indexedoptionarray.get());
    return raw->reverse_merge(other);
  }

  const ContentPtr
  UnmaskedArray::merge(const ContentPtr& other) const {
    return toIndexedOptionArray64().get()->merge(other);
  }

  const SliceItemPtr
  UnmaskedArray::asslice() const {
    return content_.get()->asslice();
  }

  const ContentPtr
  UnmaskedArray::fillna(const ContentPtr& value) const {
    return content_.get()->fillna(value);
  }

  const ContentPtr
  UnmaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (posaxis == depth + 1) {
      return content_.get()->rpad(target, posaxis, depth);
    }
    else {
      return std::make_shared<UnmaskedArray>(
        Identities::none(),
        parameters_,
        content_.get()->rpad(target, posaxis, depth));
    }
  }

  const ContentPtr
  UnmaskedArray::rpad_and_clip(int64_t target,
                               int64_t axis,
                               int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (posaxis == depth + 1) {
      return content_.get()->rpad_and_clip(target, posaxis, depth);
    }
    else {
      return std::make_shared<UnmaskedArray>(
        Identities::none(),
        parameters_,
        content_.get()->rpad_and_clip(target, posaxis, depth));
    }
  }

  const ContentPtr
  UnmaskedArray::reduce_next(const Reducer& reducer,
                             int64_t negaxis,
                             const Index64& starts,
                             const Index64& shifts,
                             const Index64& parents,
                             int64_t outlength,
                             bool mask,
                             bool keepdims) const {
    return content_.get()->reduce_next(reducer,
                                       negaxis,
                                       starts,
                                       shifts,
                                       parents,
                                       outlength,
                                       mask,
                                       keepdims);
  }

  const ContentPtr
  UnmaskedArray::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else {
      return std::make_shared<UnmaskedArray>(
        identities_,
        util::Parameters(),
        content_.get()->localindex(posaxis, depth));
    }
  }

  const ContentPtr
  UnmaskedArray::combinations(int64_t n,
                              bool replacement,
                              const util::RecordLookupPtr& recordlookup,
                              const util::Parameters& parameters,
                              int64_t axis,
                              int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument(
        std::string("in combinations, 'n' must be at least 1")
        + FILENAME(__LINE__));
    }
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }
    else {
      return std::make_shared<UnmaskedArray>(
        identities_,
        util::Parameters(),
        content_.get()->combinations(n,
                                     replacement,
                                     recordlookup,
                                     parameters,
                                     posaxis,
                                     depth));
    }
  }

  const ContentPtr
  UnmaskedArray::sort_next(int64_t negaxis,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength,
                           bool ascending,
                           bool stable,
                           bool keepdims) const {
    std::shared_ptr<Content> out = content_.get()->sort_next(negaxis,
                                                             starts,
                                                             parents,
                                                             outlength,
                                                             ascending,
                                                             stable,
                                                             keepdims);
    if (RegularArray* raw = dynamic_cast<RegularArray*>(out.get())) {
      std::shared_ptr<Content> wrapped = std::make_shared<UnmaskedArray>(
          Identities::none(),
          parameters_,
          raw->content());
      return std::make_shared<RegularArray>(
          raw->identities(),
          raw->parameters(),
          wrapped,
          raw->size());
    }
    else {
      return out;
    }
  }

  const ContentPtr
  UnmaskedArray::argsort_next(int64_t negaxis,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength,
                              bool ascending,
                              bool stable,
                              bool keepdims) const {
    std::shared_ptr<Content> out = content_.get()->argsort_next(negaxis,
                                                                starts,
                                                                parents,
                                                                outlength,
                                                                ascending,
                                                                stable,
                                                                keepdims);
    if (RegularArray* raw = dynamic_cast<RegularArray*>(out.get())) {
      std::shared_ptr<Content> wrapped = std::make_shared<UnmaskedArray>(
          Identities::none(),
          parameters_,
          raw->content());
      return std::make_shared<RegularArray>(
          raw->identities(),
          raw->parameters(),
          wrapped,
          raw->size());
    }
    else {
      return out;
    }
  }

  const ContentPtr
  UnmaskedArray::getitem_next(const SliceAt& at,
                              const Slice& tail,
                              const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnmaskedArray::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  UnmaskedArray::getitem_next(const SliceRange& range,
                              const Slice& tail,
                              const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnmaskedArray::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  UnmaskedArray::getitem_next(const SliceArray64& array,
                              const Slice& tail,
                              const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnmaskedArray::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  UnmaskedArray::getitem_next(const SliceJagged64& jagged,
                              const Slice& tail,
                              const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnmaskedArray::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  UnmaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                     const Index64& slicestops,
                                     const SliceArray64& slicecontent,
                                     const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts,
                                                     slicestops,
                                                     slicecontent,
                                                     tail);
  }

  const ContentPtr
  UnmaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                     const Index64& slicestops,
                                     const SliceMissing64& slicecontent,
                                     const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
  }

  const ContentPtr
  UnmaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                     const Index64& slicestops,
                                     const SliceJagged64& slicecontent,
                                     const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts,
                                                      slicestops,
                                                      slicecontent,
                                                      tail);
  }

  const ContentPtr
  UnmaskedArray::copy_to(kernel::lib ptr_lib) const {
    ContentPtr content = content_.get()->copy_to(ptr_lib);
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<UnmaskedArray>(identities,
                                           parameters_,
                                           content);
  }

  const ContentPtr
  UnmaskedArray::numbers_to_type(const std::string& name) const {
    ContentPtr content = content_.get()->numbers_to_type(name);
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<UnmaskedArray>(identities, parameters_, content);
  }

  template <typename S>
  const ContentPtr
  UnmaskedArray::getitem_next_jagged_generic(const Index64& slicestarts,
                                             const Index64& slicestops,
                                             const S& slicecontent,
                                             const Slice& tail) const {
    UnmaskedArray out2(identities_,
                       parameters_,
                       content_.get()->getitem_next_jagged(slicestarts,
                                                           slicestops,
                                                           slicecontent,
                                                           tail));
    return out2.simplify_optiontype();
  }

}
