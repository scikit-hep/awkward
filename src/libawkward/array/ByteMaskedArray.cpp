// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/cpu-kernels/operations.h"
#include "awkward/cpu-kernels/reducers.h"
#include "awkward/type/OptionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/None.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/VirtualArray.h"

#include "awkward/array/ByteMaskedArray.h"

namespace awkward {
  ////////// ByteMaskedForm

  ByteMaskedForm::ByteMaskedForm(bool has_identities,
                                 const util::Parameters& parameters,
                                 Index::Form mask,
                                 const FormPtr& content,
                                 bool valid_when)
      : Form(has_identities, parameters)
      , mask_(mask)
      , content_(content)
      , valid_when_(valid_when) { }

  Index::Form
  ByteMaskedForm::mask() const {
    return mask_;
  }

  const FormPtr
  ByteMaskedForm::content() const {
    return content_;
  }

  bool
  ByteMaskedForm::valid_when() const {
    return valid_when_;
  }

  const TypePtr
  ByteMaskedForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<OptionType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               content_.get()->type(typestrs));
  }

  void
  ByteMaskedForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("ByteMaskedArray");
    builder.field("mask");
    builder.string(Index::form2str(mask_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    builder.field("valid_when");
    builder.boolean(valid_when_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  ByteMaskedForm::shallow_copy() const {
    return std::make_shared<ByteMaskedForm>(has_identities_,
                                            parameters_,
                                            mask_,
                                            content_,
                                            valid_when_);
  }

  const std::string
  ByteMaskedForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  ByteMaskedForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  ByteMaskedForm::purelist_depth() const {
    return content_.get()->purelist_depth();
  }

  const std::pair<int64_t, int64_t>
  ByteMaskedForm::minmax_depth() const {
    return content_.get()->minmax_depth();
  }

  const std::pair<bool, int64_t>
  ByteMaskedForm::branch_depth() const {
    return content_.get()->branch_depth();
  }

  int64_t
  ByteMaskedForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  ByteMaskedForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  ByteMaskedForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  ByteMaskedForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  ByteMaskedForm::keys() const {
    return content_.get()->keys();
  }

  bool
  ByteMaskedForm::equal(const FormPtr& other,
                        bool check_identities,
                        bool check_parameters,
                        bool compatibility_check) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (ByteMaskedForm* t = dynamic_cast<ByteMaskedForm*>(other.get())) {
      return (mask_ == t->mask()  &&
              content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    compatibility_check)  &&
              valid_when_ == t->valid_when());
    }
    else {
      return false;
    }
  }

  ////////// ByteMaskedArray

  ByteMaskedArray::ByteMaskedArray(const IdentitiesPtr& identities,
                                   const util::Parameters& parameters,
                                   const Index8& mask,
                                   const ContentPtr& content,
                                   bool valid_when)
      : Content(identities, parameters)
      , mask_(mask)
      , content_(content)
      , valid_when_(valid_when != 0) {
    if (content.get()->length() < mask.length()) {
      throw std::invalid_argument(
        "ByteMaskedArray content must not be shorter than its mask");
    }
  }

  const Index8
  ByteMaskedArray::mask() const {
    return mask_;
  }

  const ContentPtr
  ByteMaskedArray::content() const {
    return content_;
  }

  bool
  ByteMaskedArray::valid_when() const {
    return valid_when_;
  }

  const ContentPtr
  ByteMaskedArray::project() const {
    int64_t numnull;
    struct Error err1 = kernel::bytemaskedarray_numnull(
      &numnull,
      mask_.ptr().get(),
      mask_.offset(),
      length(),
      valid_when_);
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextcarry(length() - numnull);
    struct Error err2 = kernel::bytemaskedarray_getitem_nextcarry_64(
      nextcarry.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      length(),
      valid_when_);
    util::handle_error(err2, classname(), identities_.get());

    return content_.get()->carry(nextcarry);
  }

  const ContentPtr
  ByteMaskedArray::project(const Index8& mask) const {
    if (length() != mask.length()) {
      throw std::invalid_argument(
        std::string("mask length (") + std::to_string(mask.length())
        + std::string(") is not equal to ") + classname()
        + std::string(" length (") + std::to_string(length())
        + std::string(")"));
    }

    Index8 nextmask(length());
    struct Error err = kernel::bytemaskedarray_overlay_mask8(
      nextmask.ptr().get(),
      mask.ptr().get(),
      mask.offset(),
      mask_.ptr().get(),
      mask_.offset(),
      length(),
      valid_when_);
    util::handle_error(err, classname(), identities_.get());

    //                                                       valid_when=false
    ByteMaskedArray next(identities_, parameters_, nextmask, content_, false);
    return next.project();
  }

  const Index8
  ByteMaskedArray::bytemask() const {
    if (!valid_when_) {
      return mask_;
    }
    else {
      Index8 out(length());
      struct Error err = kernel::bytemaskedarray_mask8(
        out.ptr().get(),
        mask_.ptr().get(),
        mask_.offset(),
        mask_.length(),
        valid_when_);
      util::handle_error(err, classname(), identities_.get());
      return out;
    }
  }

  const ContentPtr
  ByteMaskedArray::simplify_optiontype() const {
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

  const std::shared_ptr<IndexedOptionArray64>
  ByteMaskedArray::toIndexedOptionArray64() const {
    Index64 index(length());
    struct Error err = kernel::bytemaskedarray_toindexedarray_64(
      index.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      valid_when_);
    util::handle_error(err, classname(), identities_.get());
    return std::make_shared<IndexedOptionArray64>(identities_,
                                                  parameters_,
                                                  index,
                                                  content_);
  }

  const std::string
  ByteMaskedArray::classname() const {
    return "ByteMaskedArray";
  }

  void
  ByteMaskedArray::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() == nullptr) {
      content_.get()->setidentities(identities);
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(
          failure("content and its identities must have the same length",
                  kSliceNone,
                  kSliceNone),
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
        struct Error err = kernel::identities_extend<int32_t>(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
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
        struct Error err = kernel::identities_extend<int64_t>(
          rawsubidentities->ptr().get(),
          rawidentities->ptr().get(),
          rawidentities->offset(),
          rawidentities->length(),
          content_.get()->length());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else {
        throw std::runtime_error("unrecognized Identities specialization");
      }
    }
    identities_ = identities;
  }

  void
  ByteMaskedArray::setidentities() {
    if (length() <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       length());
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_identities<int32_t>(rawidentities->ptr().get(),
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
      struct Error err =
        kernel::new_identities<int64_t>(rawidentities->ptr().get(), length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  const TypePtr
  ByteMaskedArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  ByteMaskedArray::form(bool materialize) const {
    return std::make_shared<ByteMaskedForm>(identities_.get() != nullptr,
                                            parameters_,
                                            mask_.form(),
                                            content_.get()->form(materialize),
                                            valid_when_);
  }

  bool
  ByteMaskedArray::has_virtual_form() const {
    return content_.get()->has_virtual_form();
  }

  bool
  ByteMaskedArray::has_virtual_length() const {
    return content_.get()->has_virtual_length();
  }

  const std::string
  ByteMaskedArray::tostring_part(const std::string& indent,
                                 const std::string& pre,
                                 const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " valid_when=\""
        << (valid_when_ ? "true" : "false") << "\">\n";
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
  ByteMaskedArray::tojson_part(ToJson& builder,
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
  ByteMaskedArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    mask_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t
  ByteMaskedArray::length() const {
    return mask_.length();
  }

  const ContentPtr
  ByteMaskedArray::shallow_copy() const {
    return std::make_shared<ByteMaskedArray>(identities_,
                                             parameters_,
                                             mask_,
                                             content_,
                                             valid_when_);
  }

  const ContentPtr
  ByteMaskedArray::deep_copy(bool copyarrays,
                             bool copyindexes,
                             bool copyidentities) const {
    Index8 mask = copyindexes ? mask_.deep_copy() : mask_;
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<ByteMaskedArray>(identities,
                                             parameters_,
                                             mask,
                                             content,
                                             valid_when_);
  }

  void
  ByteMaskedArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < length()) {
      util::handle_error(
        failure("len(identities) < len(array)", kSliceNone, kSliceNone),
        identities_.get()->classname(),
        nullptr);
    }
  }

  const ContentPtr
  ByteMaskedArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const ContentPtr
  ByteMaskedArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(
        failure("index out of range", kSliceNone, at),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  ByteMaskedArray::getitem_at_nowrap(int64_t at) const {
    bool msk = (mask_.getitem_at_nowrap(at) != 0);
    if (msk == valid_when_) {
      return content_.get()->getitem_at_nowrap(at);
    }
    else {
      return none;
    }
  }

  const ContentPtr
  ByteMaskedArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range", kSliceNone, stop),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const ContentPtr
  ByteMaskedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<ByteMaskedArray>(
      identities,
      parameters_,
      mask_.getitem_range_nowrap(start, stop),
      content_.get()->getitem_range_nowrap(start, stop),
      valid_when_);
  }

  const ContentPtr
  ByteMaskedArray::getitem_field(const std::string& key) const {
    return std::make_shared<ByteMaskedArray>(
      identities_,
      util::Parameters(),
      mask_,
      content_.get()->getitem_field(key),
      valid_when_);
  }

  const ContentPtr
  ByteMaskedArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<ByteMaskedArray>(
      identities_,
      util::Parameters(),
      mask_,
      content_.get()->getitem_fields(keys),
      valid_when_);
  }

  const ContentPtr
  ByteMaskedArray::getitem_next(const SliceItemPtr& head,
                                const Slice& tail,
                                const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||
             dynamic_cast<SliceRange*>(head.get())  ||
             dynamic_cast<SliceArray64*>(head.get())  ||
             dynamic_cast<SliceJagged64*>(head.get())) {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);

      ContentPtr out = next.get()->getitem_next(head, tail, advanced);
      IndexedOptionArray64 out2(identities_, parameters_, outindex, out);
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
      throw std::runtime_error("unrecognized slice type");
    }
  }

  const ContentPtr
  ByteMaskedArray::carry(const Index64& carry) const {
    Index8 nextmask(carry.length());
    struct Error err = kernel::bytemaskedarray_getitem_carry_64(
      nextmask.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      carry.ptr().get(),
      carry.length());
    util::handle_error(err, classname(), identities_.get());
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<ByteMaskedArray>(identities,
                                             parameters_,
                                             nextmask,
                                             content_.get()->carry(carry),
                                             valid_when_);
  }

  int64_t
  ByteMaskedArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  ByteMaskedArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  ByteMaskedArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  ByteMaskedArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  ByteMaskedArray::keys() const {
    return content_.get()->keys();
  }

  const std::string
  ByteMaskedArray::validityerror(const std::string& path) const {
    return content_.get()->validityerror(path + std::string(".content"));
  }

  const ContentPtr
  ByteMaskedArray::shallow_simplify() const {
    return simplify_optiontype();
  }

  const ContentPtr
  ByteMaskedArray::num(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);

      ContentPtr out = next.get()->num(axis, depth);
      IndexedOptionArray64 out2(Identities::none(),
                                util::Parameters(),
                                outindex,
                                out);
      return out2.simplify_optiontype();
    }
  }

  const std::pair<Index64, ContentPtr>
  ByteMaskedArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      throw std::invalid_argument("axis=0 not allowed for flatten");
    }
    else {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);

      std::pair<Index64, ContentPtr> offsets_flattened =
        next.get()->offsets_and_flattened(axis, depth);
      Index64 offsets = offsets_flattened.first;
      ContentPtr flattened = offsets_flattened.second;

      if (offsets.length() == 0) {
        return std::pair<Index64, ContentPtr>(
          offsets,
          std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                 util::Parameters(),
                                                 outindex,
                                                 flattened));
      }
      else {
        Index64 outoffsets(offsets.length() + numnull);
        struct Error err =
          kernel::indexedarray_flatten_none2empty_64<int64_t>(
          outoffsets.ptr().get(),
          outindex.ptr().get(),
          outindex.offset(),
          outindex.length(),
          offsets.ptr().get(),
          offsets.offset(),
          offsets.length());
        util::handle_error(err, classname(), identities_.get());
        return std::pair<Index64, ContentPtr>(outoffsets, flattened);
      }
    }
  }

  bool
  ByteMaskedArray::mergeable(const ContentPtr& other, bool mergebool) const {
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
  ByteMaskedArray::reverse_merge(const ContentPtr& other) const {
    ContentPtr indexedoptionarray = toIndexedOptionArray64();
    IndexedOptionArray64* raw =
      dynamic_cast<IndexedOptionArray64*>(indexedoptionarray.get());
    return raw->reverse_merge(other);
  }

  const ContentPtr
  ByteMaskedArray::merge(const ContentPtr& other) const {
    return toIndexedOptionArray64().get()->merge(other);
  }

  const SliceItemPtr
  ByteMaskedArray::asslice() const {
    return toIndexedOptionArray64().get()->asslice();
  }

  const ContentPtr
  ByteMaskedArray::fillna(const ContentPtr& value) const {
    return toIndexedOptionArray64().get()->fillna(value);
  }

  const ContentPtr
  ByteMaskedArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (toaxis == depth + 1) {
      Index8 mask = bytemask();
      Index64 index(mask.length());
      struct Error err =
        kernel::IndexedOptionArray_rpad_and_clip_mask_axis1_64(
        index.ptr().get(),
        mask.ptr().get(),
        mask.length());
      util::handle_error(err, classname(), identities_.get());

      ContentPtr next = project().get()->rpad(target, toaxis, depth);
      return std::make_shared<IndexedOptionArray64>(
        Identities::none(),
        util::Parameters(),
        index,
        next).get()->simplify_optiontype();
    }
    else {
      return std::make_shared<ByteMaskedArray>(
        Identities::none(),
        parameters_,
        mask_,
        content_.get()->rpad(target, toaxis, depth),
        valid_when_);
    }
  }

  const ContentPtr
  ByteMaskedArray::rpad_and_clip(int64_t target,
                                 int64_t axis,
                                 int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return rpad_axis0(target, true);
    }
    else if (toaxis == depth + 1) {
      Index8 mask = bytemask();
      Index64 index(mask.length());
      struct Error err =
        kernel::IndexedOptionArray_rpad_and_clip_mask_axis1_64(
        index.ptr().get(),
        mask.ptr().get(),
        mask.length());
      util::handle_error(err, classname(), identities_.get());

      ContentPtr next = project().get()->rpad_and_clip(target, toaxis, depth);
      return std::make_shared<IndexedOptionArray64>(
        Identities::none(),
        util::Parameters(),
        index,
        next).get()->simplify_optiontype();
    }
    else {
      return std::make_shared<ByteMaskedArray>(
        Identities::none(),
        parameters_,
        mask_,
        content_.get()->rpad_and_clip(target, toaxis, depth),
        valid_when_);
    }
  }

  const ContentPtr
  ByteMaskedArray::reduce_next(const Reducer& reducer,
                               int64_t negaxis,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength,
                               bool mask,
                               bool keepdims) const {
    int64_t numnull;
    struct Error err1 = kernel::bytemaskedarray_numnull(
      &numnull,
      mask_.ptr().get(),
      mask_.offset(),
      length(),
      valid_when_);
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextparents(length() - numnull);
    Index64 nextcarry(length() - numnull);
    Index64 outindex(length());
    struct Error err2 = kernel::bytemaskedarray_reduce_next_64(
      nextcarry.ptr().get(),
      nextparents.ptr().get(),
      outindex.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      parents.ptr().get(),
      parents.offset(),
      length(),
      valid_when_);
    util::handle_error(err2, classname(), identities_.get());

    ContentPtr next = content_.get()->carry(nextcarry);
    ContentPtr out = next.get()->reduce_next(reducer,
                                             negaxis,
                                             starts,
                                             nextparents,
                                             outlength,
                                             mask,
                                             keepdims);

    std::pair<bool, int64_t> branchdepth = branch_depth();
    if (!branchdepth.first  &&  negaxis == branchdepth.second) {
      return out;
    }
    else {
      if (RegularArray* raw =
          dynamic_cast<RegularArray*>(out.get())) {
        out = raw->toListOffsetArray64(true);
      }
      if (ListOffsetArray64* raw =
          dynamic_cast<ListOffsetArray64*>(out.get())) {
        Index64 outoffsets(starts.length() + 1);
        if (starts.length() > 0  &&  starts.getitem_at_nowrap(0) != 0) {
          throw std::runtime_error(
            "reduce_next with unbranching depth > negaxis expects "
            "a ListOffsetArray64 whose offsets start at zero");
        }
        struct Error err3 = kernel::indexedarray_reduce_next_fix_offsets_64(
          outoffsets.ptr().get(),
          starts.ptr().get(),
          starts.offset(),
          starts.length(),
          outindex.length());
        util::handle_error(err3, classname(), identities_.get());

        return std::make_shared<ListOffsetArray64>(
          raw->identities(),
          raw->parameters(),
          outoffsets,
          std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                 util::Parameters(),
                                                 outindex,
                                                 raw->content()));
      }
      else {
        throw std::runtime_error(
          std::string("reduce_next with unbranching depth > negaxis is only "
                      "expected to return RegularArray or ListOffsetArray64; "
                      "instead, it returned ")
          + out.get()->classname());
      }
    }
  }

  const ContentPtr
  ByteMaskedArray::localindex(int64_t axis, int64_t depth) const {
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (toaxis == depth) {
      return localindex_axis0();
    }
    else {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);
      ContentPtr out = next.get()->localindex(axis, depth);
      IndexedOptionArray64 out2(Identities::none(),
                                util::Parameters(),
                                outindex,
                                out);
      return out2.simplify_optiontype();
    }
  }

  const ContentPtr
  ByteMaskedArray::combinations(int64_t n,
                                bool replacement,
                                const util::RecordLookupPtr& recordlookup,
                                const util::Parameters& parameters,
                                int64_t axis,
                                int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument("in combinations, 'n' must be at least 1");
    }
    int64_t toaxis = axis_wrap_if_negative(axis);
    if (axis == depth) {
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }
    else {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);
      ContentPtr out = next.get()->combinations(n,
                                                replacement,
                                                recordlookup,
                                                parameters,
                                                axis,
                                                depth);
      IndexedOptionArray64 out2(Identities::none(),
                                util::Parameters(),
                                outindex,
                                out);
      return out2.simplify_optiontype();
    }
  }

  const ContentPtr
  ByteMaskedArray::getitem_next(const SliceAt& at,
                                const Slice& tail,
                                const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: ByteMaskedArray::getitem_next(at)");
  }

  const ContentPtr
  ByteMaskedArray::getitem_next(const SliceRange& range,
                                const Slice& tail,
                                const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: ByteMaskedArray::getitem_next(range)");
  }

  const ContentPtr
  ByteMaskedArray::getitem_next(const SliceArray64& array,
                                const Slice& tail,
                                const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: ByteMaskedArray::getitem_next(array)");
  }

  const ContentPtr
  ByteMaskedArray::getitem_next(const SliceJagged64& jagged,
                                const Slice& tail,
                                const Index64& advanced) const {
    throw std::runtime_error(
      "undefined operation: ByteMaskedArray::getitem_next(jagged)");
  }

  const ContentPtr
  ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                       const Index64& slicestops,
                                       const SliceArray64& slicecontent,
                                       const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts,
                                                     slicestops,
                                                     slicecontent,
                                                     tail);
  }

  const ContentPtr
  ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                       const Index64& slicestops,
                                       const SliceMissing64& slicecontent,
                                       const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
  }

  const ContentPtr
  ByteMaskedArray::getitem_next_jagged(const Index64& slicestarts,
                                       const Index64& slicestops,
                                       const SliceJagged64& slicecontent,
                                       const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts,
                                                      slicestops,
                                                      slicecontent,
                                                      tail);
  }

  template <typename S>
  const ContentPtr
  ByteMaskedArray::getitem_next_jagged_generic(const Index64& slicestarts,
                                               const Index64& slicestops,
                                               const S& slicecontent,
                                               const Slice& tail) const {
      int64_t numnull;
      std::pair<Index64, Index64> pair = nextcarry_outindex(numnull);
      Index64 nextcarry = pair.first;
      Index64 outindex = pair.second;

      ContentPtr next = content_.get()->carry(nextcarry);
      ContentPtr out = next.get()->getitem_next_jagged(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
      IndexedOptionArray64 out2(identities_, parameters_, outindex, out);
      return out2.simplify_optiontype();
  }

  const std::pair<Index64, Index64>
  ByteMaskedArray::nextcarry_outindex(int64_t& numnull) const {
    struct Error err1 = kernel::bytemaskedarray_numnull(
      &numnull,
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      valid_when_);
    util::handle_error(err1, classname(), identities_.get());

    Index64 nextcarry(length() - numnull);
    Index64 outindex(length());
    struct Error err2 = kernel::bytemaskedarray_getitem_nextcarry_outindex<int64_t>(
      nextcarry.ptr().get(),
      outindex.ptr().get(),
      mask_.ptr().get(),
      mask_.offset(),
      mask_.length(),
      valid_when_);
    util::handle_error(err2, classname(), identities_.get());

    return std::pair<Index64, Index64>(nextcarry, outindex);
  }

}
