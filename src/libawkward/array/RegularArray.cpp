// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/RegularArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/RegularArray.cpp", line)

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/type/RegularType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"

#include "awkward/array/RegularArray.h"

namespace awkward {
  ////////// RegularForm

  RegularForm::RegularForm(bool has_identities,
                           const util::Parameters& parameters,
                           const FormKey& form_key,
                           const FormPtr& content,
                           int64_t size)
      : Form(has_identities, parameters, form_key)
      , content_(content)
      , size_(size) { }

  const FormPtr
  RegularForm::content() const {
    return content_;
  }

  int64_t
  RegularForm::size() const {
    return size_;
  }

  const TypePtr
  RegularForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<RegularType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               content_.get()->type(typestrs),
               size_);
  }

  void
  RegularForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("RegularArray");
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    builder.field("size");
    builder.integer(size_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  RegularForm::shallow_copy() const {
    return std::make_shared<RegularForm>(has_identities_,
                                         parameters_,
                                         form_key_,
                                         content_,
                                         size_);
  }

  const std::string
  RegularForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  RegularForm::purelist_isregular() const {
    return content_.get()->purelist_isregular();
  }

  int64_t
  RegularForm::purelist_depth() const {
    return content_.get()->purelist_depth() + 1;
  }

  const std::pair<int64_t, int64_t>
  RegularForm::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1,
                                       content_depth.second + 1);
  }

  const std::pair<bool, int64_t>
  RegularForm::branch_depth() const {
    std::pair<bool, int64_t> content_depth = content_.get()->branch_depth();
    return std::pair<bool, int64_t>(content_depth.first,
                                    content_depth.second + 1);
  }

  int64_t
  RegularForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  RegularForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  RegularForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  RegularForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  RegularForm::keys() const {
    return content_.get()->keys();
  }

  bool
  RegularForm::equal(const FormPtr& other,
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
    if (RegularForm* t = dynamic_cast<RegularForm*>(other.get())) {
      return (content_.get()->equal(t->content(),
                                    check_identities,
                                    check_parameters,
                                    check_form_key,
                                    compatibility_check)  &&
              size_ == t->size());
    }
    else {
      return false;
    }
  }

  const FormPtr
  RegularForm::getitem_field(const std::string& key) const {
    return content_.get()->getitem_field(key);
  }

  ////////// RegularArray

  RegularArray::RegularArray(const IdentitiesPtr& identities,
                             const util::Parameters& parameters,
                             const ContentPtr& content,
                             int64_t size)
      : Content(identities, parameters)
      , content_(content)
      , size_(size) {
    if (size < 0) {
      throw std::invalid_argument(
        std::string("RegularArray size must be non-negative")
        + FILENAME(__LINE__));
    }
  }

  const ContentPtr
  RegularArray::content() const {
    return content_;
  }

  int64_t
  RegularArray::size() const {
    return size_;
  }

  Index64
  RegularArray::compact_offsets64(bool start_at_zero) const {
    int64_t len = length();
    Index64 out(len + 1);
    struct Error err = kernel::RegularArray_compact_offsets_64(
      kernel::lib::cpu,   // DERIVE
      out.data(),
      len,
      size_);
    util::handle_error(err, classname(), identities_.get());
    return out;
  }

  const ContentPtr
  RegularArray::broadcast_tooffsets64(const Index64& offsets) const {
    if (offsets.length() == 0  ||  offsets.getitem_at_nowrap(0) != 0) {
      throw std::invalid_argument(
        std::string("broadcast_tooffsets64 can only be used with offsets that start at 0")
        + FILENAME(__LINE__));
    }

    int64_t len = length();
    if (offsets.length() - 1 != len) {
      throw std::invalid_argument(
        std::string("cannot broadcast RegularArray of length ")
        + std::to_string(len) + (" to length ")
        + std::to_string(offsets.length() - 1) + FILENAME(__LINE__));
    }

    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      identities =
        identities_.get()->getitem_range_nowrap(0, offsets.length() - 1);
    }

    if (size_ == 1) {
      int64_t carrylen = offsets.getitem_at_nowrap(offsets.length() - 1);
      Index64 nextcarry(carrylen);
      struct Error err = kernel::RegularArray_broadcast_tooffsets_size1_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        offsets.data(),
        offsets.length());
      util::handle_error(err, classname(), identities_.get());
      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
      return std::make_shared<ListOffsetArray64>(identities,
                                                 parameters_,
                                                 offsets,
                                                 nextcontent);
    }
    else {
      struct Error err = kernel::RegularArray_broadcast_tooffsets_64(
        kernel::lib::cpu,   // DERIVE
        offsets.data(),
        offsets.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<ListOffsetArray64>(identities,
                                                 parameters_,
                                                 offsets,
                                                 content_);
    }
  }

  const ContentPtr
  RegularArray::toRegularArray() const {
    return shallow_copy();
  }

  const ContentPtr
  RegularArray::toListOffsetArray64(bool start_at_zero) const {
    Index64 offsets = compact_offsets64(start_at_zero);
    return broadcast_tooffsets64(offsets);
  }

  const std::string
  RegularArray::classname() const {
    return "RegularArray";
  }

  void
  RegularArray::setidentities(const IdentitiesPtr& identities) {
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
      IdentitiesPtr bigidentities = identities;
      if (content_.get()->length() > kMaxInt32) {
        bigidentities = identities.get()->to64();
      }
      if (Identities32* rawidentities =
          dynamic_cast<Identities32*>(bigidentities.get())) {
        IdentitiesPtr subidentities =
          std::make_shared<Identities32>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width() + 1,
                                         content_.get()->length());
        Identities32* rawsubidentities =
          reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = kernel::Identities_from_RegularArray<int32_t>(
          kernel::lib::cpu,   // DERIVE
          rawsubidentities->data(),
          rawidentities->data(),
          size_,
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        content_.get()->setidentities(subidentities);
      }
      else if (Identities64* rawidentities =
               dynamic_cast<Identities64*>(bigidentities.get())) {
        IdentitiesPtr subidentities =
          std::make_shared<Identities64>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width() + 1,
                                         content_.get()->length());
        Identities64* rawsubidentities =
          reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = kernel::Identities_from_RegularArray<int64_t>(
          kernel::lib::cpu,   // DERIVE
          rawsubidentities->data(),
          rawidentities->data(),
          size_,
          content_.get()->length(),
          length(),
          rawidentities->width());
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
  RegularArray::setidentities() {
    if (length() < kMaxInt32) {
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
  RegularArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  RegularArray::form(bool materialize) const {
    return std::make_shared<RegularForm>(identities_.get() != nullptr,
                                         parameters_,
                                         FormKey(nullptr),
                                         content_.get()->form(materialize),
                                         size_);
  }

  bool
  RegularArray::has_virtual_form() const {
    return content_.get()->has_virtual_form();
  }

  bool
  RegularArray::has_virtual_length() const {
    return content_.get()->has_virtual_length();
  }

  const std::string
  RegularArray::tostring_part(const std::string& indent,
                              const std::string& pre,
                              const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " size=\"" << size_
        << "\">\n";
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
  RegularArray::tojson_part(ToJson& builder, bool include_beginendlist) const {
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
  RegularArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t
  RegularArray::length() const {
    return (size_ == 0
              ? 0
              : content_.get()->length() / size_);   // floor of length / size
  }

  const ContentPtr
  RegularArray::shallow_copy() const {
    return std::make_shared<RegularArray>(identities_,
                                          parameters_,
                                          content_,
                                          size_);
  }

  const ContentPtr
  RegularArray::deep_copy(bool copyarrays,
                          bool copyindexes,
                          bool copyidentities) const {
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<RegularArray>(identities,
                                          parameters_,
                                          content,
                                          size_);
  }

  void
  RegularArray::check_for_iteration() const {
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
  RegularArray::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  const ContentPtr
  RegularArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    int64_t len = length();
    if (regular_at < 0) {
      regular_at += len;
    }
    if (!(0 <= regular_at  &&  regular_at < len)) {
      util::handle_error(
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  RegularArray::getitem_at_nowrap(int64_t at) const {
    return content_.get()->getitem_range_nowrap(at*size_, (at + 1)*size_);
  }

  const ContentPtr
  RegularArray::getitem_range(int64_t start, int64_t stop) const {
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
  RegularArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<RegularArray>(
      identities_,
      parameters_,
      content_.get()->getitem_range_nowrap(start*size_, stop*size_), size_);
  }

  const ContentPtr
  RegularArray::getitem_field(const std::string& key) const {
    return std::make_shared<RegularArray>(
      identities_,
      util::Parameters(),
      content_.get()->getitem_field(key), size_);
  }

  const ContentPtr
  RegularArray::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<RegularArray>(
      identities_,
      util::Parameters(),
      content_.get()->getitem_fields(keys), size_);
  }

  const ContentPtr
  RegularArray::carry(const Index64& carry, bool allow_lazy) const {
    Index64 nextcarry(carry.length()*size_);

    struct Error err = kernel::RegularArray_getitem_carry_64(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      carry.data(),
      carry.length(),
      size_);
    util::handle_error(err, classname(), identities_.get());

    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<RegularArray>(identities,
                                          parameters_,
                                          content_.get()->carry(nextcarry, allow_lazy),
                                          size_);
  }

  int64_t
  RegularArray::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  RegularArray::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  RegularArray::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  RegularArray::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  RegularArray::keys() const {
    return content_.get()->keys();
  }

  const std::string
  RegularArray::validityerror(const std::string& path) const {
    if (size_ < 1) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string("size < 1")
              + FILENAME(__LINE__));
    }
    return content_.get()->validityerror(path + std::string(".content"));
  }

  const ContentPtr
  RegularArray::shallow_simplify() const {
    return shallow_copy();
  }

  const ContentPtr
  RegularArray::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else if (posaxis == depth + 1) {
      Index64 tonum(length());
      struct Error err = kernel::RegularArray_num_64(
        tonum.ptr_lib(),
        tonum.data(),
        size_,
        length());
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<NumpyArray>(tonum);
    }
    else {
      ContentPtr next = content_.get()->num(posaxis, depth + 1);
      return std::make_shared<RegularArray>(Identities::none(),
                                            util::Parameters(),
                                            next,
                                            size_);
    }
  }

  const std::pair<Index64, ContentPtr>
  RegularArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return toListOffsetArray64(true).get()->offsets_and_flattened(axis, depth);
  }

  bool
  RegularArray::mergeable(const ContentPtr& other, bool mergebool) const {
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
    else if (IndexedArray32* rawother =
             dynamic_cast<IndexedArray32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArrayU32* rawother =
             dynamic_cast<IndexedArrayU32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedArray64* rawother =
             dynamic_cast<IndexedArray64*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray32* rawother =
             dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (IndexedOptionArray64* rawother =
             dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (ByteMaskedArray* rawother =
             dynamic_cast<ByteMaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (BitMaskedArray* rawother =
             dynamic_cast<BitMaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }
    else if (UnmaskedArray* rawother =
             dynamic_cast<UnmaskedArray*>(other.get())) {
      return mergeable(rawother->content(), mergebool);
    }

    if (RegularArray* rawother =
        dynamic_cast<RegularArray*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArray32* rawother =
             dynamic_cast<ListArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArrayU32* rawother =
             dynamic_cast<ListArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListArray64* rawother =
             dynamic_cast<ListArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArray32* rawother =
             dynamic_cast<ListOffsetArray32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArrayU32* rawother =
             dynamic_cast<ListOffsetArrayU32*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else if (ListOffsetArray64* rawother =
             dynamic_cast<ListOffsetArray64*>(other.get())) {
      return content_.get()->mergeable(rawother->content(), mergebool);
    }
    else {
      return false;
    }
  }

  const ContentPtr
  RegularArray::merge(const ContentPtr& other) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return merge(raw->array());
    }

    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

    if (dynamic_cast<EmptyArray*>(other.get())) {
      return shallow_copy();
    }
    else if (IndexedArray32* rawother =
             dynamic_cast<IndexedArray32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedArrayU32* rawother =
             dynamic_cast<IndexedArrayU32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedArray64* rawother =
             dynamic_cast<IndexedArray64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedOptionArray32* rawother =
             dynamic_cast<IndexedOptionArray32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (IndexedOptionArray64* rawother =
             dynamic_cast<IndexedOptionArray64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (ByteMaskedArray* rawother =
             dynamic_cast<ByteMaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (BitMaskedArray* rawother =
             dynamic_cast<BitMaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnmaskedArray* rawother =
             dynamic_cast<UnmaskedArray*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_32* rawother =
             dynamic_cast<UnionArray8_32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_U32* rawother =
             dynamic_cast<UnionArray8_U32*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }
    else if (UnionArray8_64* rawother =
             dynamic_cast<UnionArray8_64*>(other.get())) {
      return rawother->reverse_merge(shallow_copy());
    }

    if (RegularArray* rawother = dynamic_cast<RegularArray*>(other.get())) {
      if (size_ == rawother->size()) {
        ContentPtr mine =
          content_.get()->getitem_range_nowrap(0, size_*length());
        ContentPtr theirs =
          rawother->content().get()->getitem_range_nowrap(
            0, rawother->size()*rawother->length());
        ContentPtr content = mine.get()->merge(theirs);
        return std::make_shared<RegularArray>(Identities::none(),
                                              parameters_,
                                              content,
                                              size_);
      }
      else {
        return toListOffsetArray64(true).get()->merge(other);
      }
    }
    else if (dynamic_cast<ListArray32*>(other.get())  ||
             dynamic_cast<ListArrayU32*>(other.get())  ||
             dynamic_cast<ListArray64*>(other.get())  ||
             dynamic_cast<ListOffsetArray32*>(other.get())  ||
             dynamic_cast<ListOffsetArrayU32*>(other.get())  ||
             dynamic_cast<ListOffsetArray64*>(other.get())) {
      return toListOffsetArray64(true).get()->merge(other);
    }
    else {
      throw std::invalid_argument(
        std::string("cannot merge ") + classname() + std::string(" with ")
        + other.get()->classname() + FILENAME(__LINE__));
    }
  }

  const SliceItemPtr
  RegularArray::asslice() const {
    throw std::invalid_argument(
      std::string("slice items can have all fixed-size dimensions (to follow NumPy's "
                  "slice rules) or they can have all var-sized dimensions (for jagged "
                  "indexing), but not both in the same slice item") + FILENAME(__LINE__));
  }

  const ContentPtr
  RegularArray::fillna(const ContentPtr& value) const {
    return std::make_shared<RegularArray>(identities_,
                                          parameters_,
                                          content().get()->fillna(value),
                                          size_);
  }

  const ContentPtr
  RegularArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (posaxis == depth + 1) {
      if (target < size_) {
        return shallow_copy();
      }
      else {
        return rpad_and_clip(target, posaxis, depth);
      }
    }
    else {
      return std::make_shared<RegularArray>(
        Identities::none(),
        parameters_,
        content_.get()->rpad(target, posaxis, depth + 1),
        size_);
    }
  }

  const ContentPtr
  RegularArray::rpad_and_clip(int64_t target,
                              int64_t axis,
                              int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, true);
    }
    else if (posaxis == depth + 1) {
      Index64 index(length() * target);
      struct Error err = kernel::RegularArray_rpad_and_clip_axis1_64(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        target,
        size_,
        length());
      util::handle_error(err, classname(), identities_.get());
      std::shared_ptr<IndexedOptionArray64> next =
        std::make_shared<IndexedOptionArray64>(Identities::none(),
                                               util::Parameters(),
                                               index,
                                               content());
      return std::make_shared<RegularArray>(
        Identities::none(),
        parameters_,
        next.get()->simplify_optiontype(),
        target);
    }
    else {
      return std::make_shared<RegularArray>(
        Identities::none(),
        parameters_,
        content_.get()->rpad_and_clip(target, posaxis, depth + 1),
        size_);
    }
  }

  const ContentPtr
  RegularArray::reduce_next(const Reducer& reducer,
                            int64_t negaxis,
                            const Index64& starts,
                            const Index64& shifts,
                            const Index64& parents,
                            int64_t outlength,
                            bool mask,
                            bool keepdims) const {
    return toListOffsetArray64(true).get()->reduce_next(reducer,
                                                        negaxis,
                                                        starts,
                                                        shifts,
                                                        parents,
                                                        outlength,
                                                        mask,
                                                        keepdims);
  }

  const ContentPtr
  RegularArray::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else if (posaxis == depth + 1) {
      Index64 localindex(length()*size_);
      struct Error err = kernel::RegularArray_localindex_64(
        kernel::lib::cpu,   // DERIVE
        localindex.data(),
        size_,
        length());
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<RegularArray>(
        identities_,
        util::Parameters(),
        std::make_shared<NumpyArray>(localindex),
        size_);
    }
    else {
      return std::make_shared<RegularArray>(
        identities_,
        util::Parameters(),
        content_.get()->localindex(posaxis, depth + 1),
        size_);
    }
  }

  const ContentPtr
  RegularArray::combinations(int64_t n,
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

    else if (posaxis == depth + 1) {
      int64_t size = size_;
      if (replacement) {
        size += (n - 1);
      }
      int64_t thisn = n;
      int64_t combinationslen;
      if (thisn > size) {
        combinationslen = 0;
      }
      else if (thisn == size) {
        combinationslen = 1;
      }
      else {
        if (thisn * 2 > size) {
          thisn = size - thisn;
        }
        combinationslen = size;
        for (int64_t j = 2;  j <= thisn;  j++) {
          combinationslen *= (size - j + 1);
          combinationslen /= j;
        }
      }

      int64_t totallen = combinationslen * length();

      std::vector<std::shared_ptr<int64_t>> tocarry;
      std::vector<int64_t*> tocarryraw;
      for (int64_t j = 0;  j < n;  j++) {
        std::shared_ptr<int64_t> ptr(new int64_t[(size_t)totallen],
                                     kernel::array_deleter<int64_t>());
        tocarry.push_back(ptr);
        tocarryraw.push_back(ptr.get());
      }
      IndexOf<int64_t> toindex(size);
      IndexOf<int64_t> fromindex(size);
      struct Error err = kernel::RegularArray_combinations_64(
        kernel::lib::cpu,   // DERIVE
        tocarryraw.data(),
        toindex.data(),
        fromindex.data(),
        n,
        replacement,
        size_,
        length());
      util::handle_error(err, classname(), identities_.get());

      ContentPtrVec contents;
      for (auto ptr : tocarry) {
        contents.push_back(content_.get()->carry(
            Index64(ptr, 0, totallen, kernel::lib::cpu),   // DERIVE
        true));
      }
      ContentPtr recordarray =
        std::make_shared<RecordArray>(Identities::none(),
                                      parameters,
                                      contents,
                                      recordlookup);

      return std::make_shared<RegularArray>(identities_,
                                            util::Parameters(),
                                            recordarray,
                                            combinationslen);
    }

    else {
      ContentPtr next = content_.get()
                        ->getitem_range_nowrap(0, length()*size_).get()
                        ->combinations(n,
                                       replacement,
                                       recordlookup,
                                       parameters,
                                       posaxis,
                                       depth + 1);
      return std::make_shared<RegularArray>(identities_,
                                            util::Parameters(),
                                            next,
                                            size_);
    }
  }

  const ContentPtr
  RegularArray::sort_next(int64_t negaxis,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength,
                          bool ascending,
                          bool stable,
                          bool keepdims) const {
    std::shared_ptr<Content> out = toListOffsetArray64(true).get()->sort_next(
                                       negaxis,
                                       starts,
                                       parents,
                                       outlength,
                                       ascending,
                                       stable,
                                       keepdims);
    if (RegularArray* raw1 =
            dynamic_cast<RegularArray*>(out.get())) {
      if (ListOffsetArray64* raw2 =
              dynamic_cast<ListOffsetArray64*>(raw1->content().get())) {
        return std::make_shared<RegularArray>(
            raw1->identities(),
            raw1->parameters(),
            raw2->toRegularArray(),
            raw1->size());
      }
    }
    return out;
  }

  const ContentPtr
  RegularArray::argsort_next(int64_t negaxis,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength,
                             bool ascending,
                             bool stable,
                             bool keepdims) const {
    std::shared_ptr<Content> out = toListOffsetArray64(true).get()->argsort_next(
                                       negaxis,
                                       starts,
                                       parents,
                                       outlength,
                                       ascending,
                                       stable,
                                       keepdims);
    if (RegularArray* raw1 =
            dynamic_cast<RegularArray*>(out.get())) {
      if (ListOffsetArray64* raw2 =
              dynamic_cast<ListOffsetArray64*>(raw1->content().get())) {
        return std::make_shared<RegularArray>(
            raw1->identities(),
            raw1->parameters(),
            raw2->toRegularArray(),
            raw1->size());
      }
    }
    return out;
  }

  const ContentPtr
  RegularArray::getitem_next(const SliceAt& at,
                             const Slice& tail,
                             const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::runtime_error(
        std::string("RegularArray::getitem_next(SliceAt): advanced.length() != 0")
        + FILENAME(__LINE__));
    }
    int64_t len = length();
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 nextcarry(len);

    struct Error err = kernel::RegularArray_getitem_next_at_64(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      at.at(),
      len,
      size_);
    util::handle_error(err, classname(), identities_.get());

    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
    return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
  }

  const ContentPtr
  RegularArray::getitem_next(const SliceRange& range,
                             const Slice& tail,
                             const Index64& advanced) const {
    int64_t len = length();
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();

    if (range.step() == 0) {
      throw std::runtime_error(
        std::string("RegularArray::getitem_next(SliceRange): range.step() == 0")
        + FILENAME(__LINE__));
    }
    int64_t regular_start = range.start();
    int64_t regular_stop = range.stop();
    int64_t regular_step = std::abs(range.step());
    kernel::regularize_rangeslice(&regular_start,
                                  &regular_stop,
                                  range.step() > 0,
                                  range.start() != Slice::none(),
                                  range.stop() != Slice::none(),
                                  size_);
    int64_t nextsize = 0;
    if (range.step() > 0  &&  regular_stop - regular_start > 0) {
      int64_t diff = regular_stop - regular_start;
      nextsize = diff / regular_step;
      if (diff % regular_step != 0) {
        nextsize++;
      }
    }
    else if (range.step() < 0  &&  regular_stop - regular_start < 0) {
      int64_t diff = regular_start - regular_stop;
      nextsize = diff / regular_step;
      if (diff % regular_step != 0) {
        nextsize++;
      }
    }

    Index64 nextcarry(len*nextsize);

    struct Error err = kernel::RegularArray_getitem_next_range_64(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      regular_start,
      range.step(),
      len,
      size_,
      nextsize);
    util::handle_error(err, classname(), identities_.get());

    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);

    if (advanced.length() == 0) {
      return std::make_shared<RegularArray>(
        identities_,
        parameters_,
        nextcontent.get()->getitem_next(nexthead, nexttail, advanced),
        nextsize);
    }
    else {
      Index64 nextadvanced(len*nextsize);

      struct Error err = kernel::RegularArray_getitem_next_range_spreadadvanced_64(
        kernel::lib::cpu,   // DERIVE
        nextadvanced.data(),
        advanced.data(),
        len,
        nextsize);
      util::handle_error(err, classname(), identities_.get());

      return std::make_shared<RegularArray>(
        identities_,
        parameters_,
        nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced),
        nextsize);
    }
  }

  const ContentPtr
  RegularArray::getitem_next(const SliceArray64& array,
                             const Slice& tail,
                             const Index64& advanced) const {
    int64_t len = length();
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 flathead = array.ravel();
    Index64 regular_flathead(flathead.length());

    struct Error err = kernel::RegularArray_getitem_next_array_regularize_64(
      kernel::lib::cpu,   // DERIVE
      regular_flathead.data(),
      flathead.data(),
      flathead.length(),
      size_);
    util::handle_error(err, classname(), identities_.get());

    if (advanced.length() == 0) {
      Index64 nextcarry(len*flathead.length());
      Index64 nextadvanced(len*flathead.length());

      struct Error err = kernel::RegularArray_getitem_next_array_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        regular_flathead.data(),
        len,
        regular_flathead.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());

      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);

      return getitem_next_array_wrap(
               nextcontent.get()->getitem_next(nexthead,
                                               nexttail,
                                               nextadvanced),
               array.shape());
    }
    else {
      Index64 nextcarry(len);
      Index64 nextadvanced(len);

      struct Error err = kernel::RegularArray_getitem_next_array_advanced_64(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        advanced.data(),
        regular_flathead.data(),
        len,
        regular_flathead.length(),
        size_);
      util::handle_error(err, classname(), identities_.get());

      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
      return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
    }
  }

  const ContentPtr
  RegularArray::getitem_next(const SliceJagged64& jagged,
                             const Slice& tail,
                             const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument(
        std::string("cannot mix jagged slice with NumPy-style advanced indexing")
        + FILENAME(__LINE__));
    }

    if (jagged.length() != size_) {
      throw std::invalid_argument(
        std::string("cannot fit jagged slice with length ")
        + std::to_string(jagged.length()) + std::string(" into ")
        + classname() + std::string(" of size ") + std::to_string(size_)
        + FILENAME(__LINE__));
    }

    int64_t regularlength = length();
    Index64 singleoffsets = jagged.offsets();
    Index64 multistarts(jagged.length()*regularlength);
    Index64 multistops(jagged.length()*regularlength);
    struct Error err = kernel::RegularArray_getitem_jagged_expand_64(
      kernel::lib::cpu,   // DERIVE
      multistarts.data(),
      multistops.data(),
      singleoffsets.data(),
      jagged.length(),
      regularlength);
    util::handle_error(err, classname(), identities_.get());

    ContentPtr down = content_.get()->getitem_next_jagged(multistarts,
                                                          multistops,
                                                          jagged.content(),
                                                          tail);

    return std::make_shared<RegularArray>(Identities::none(),
                                          util::Parameters(),
                                          down,
                                          jagged.length());
  }

  const ContentPtr
  RegularArray::getitem_next_jagged(const Index64& slicestarts,
                                    const Index64& slicestops,
                                    const SliceArray64& slicecontent,
                                    const Slice& tail) const {
    ContentPtr self = toListOffsetArray64(true);
    return self.get()->getitem_next_jagged(slicestarts,
                                           slicestops,
                                           slicecontent,
                                           tail);
  }

  const ContentPtr
  RegularArray::getitem_next_jagged(const Index64& slicestarts,
                                    const Index64& slicestops,
                                    const SliceMissing64& slicecontent,
                                    const Slice& tail) const {
    ContentPtr self = toListOffsetArray64(true);
    return self.get()->getitem_next_jagged(slicestarts,
                                           slicestops,
                                           slicecontent,
                                           tail);
  }

  const ContentPtr
  RegularArray::getitem_next_jagged(const Index64& slicestarts,
                                    const Index64& slicestops,
                                    const SliceJagged64& slicecontent,
                                    const Slice& tail) const {
    ContentPtr self = toListOffsetArray64(true);
    return self.get()->getitem_next_jagged(slicestarts,
                                           slicestops,
                                           slicecontent,
                                           tail);
  }

  const ContentPtr
  RegularArray::copy_to(kernel::lib ptr_lib) const {
    ContentPtr content = content_.get()->copy_to(ptr_lib);
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<RegularArray>(identities,
                                          parameters_,
                                          content,
                                          size_);
  }

  const ContentPtr
  RegularArray::numbers_to_type(const std::string& name) const {
    ContentPtr content = content_.get()->numbers_to_type(name);
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<RegularArray>(identities,
                                          parameters_,
                                          content,
                                          size_);
  }

}
