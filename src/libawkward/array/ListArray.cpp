// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/ListArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/ListArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/type/ListType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/VirtualArray.h"

#define AWKWARD_LISTARRAY_NO_EXTERN_TEMPLATE
#include "awkward/array/ListArray.h"
#include "awkward/common.h"

namespace awkward {
  ////////// ListForm

  ListForm::ListForm(bool has_identities,
                     const util::Parameters& parameters,
                     const FormKey& form_key,
                     Index::Form starts,
                     Index::Form stops,
                     const FormPtr& content)
      : Form(has_identities, parameters, form_key)
      , starts_(starts)
      , stops_(stops)
      , content_(content) { }

  Index::Form
  ListForm::starts() const {
    return starts_;
  }

  Index::Form
  ListForm::stops() const {
    return stops_;
  }

  const FormPtr
  ListForm::content() const {
    return content_;
  }

  const TypePtr
  ListForm::type(const util::TypeStrs& typestrs) const {
    return std::make_shared<ListType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               content_.get()->type(typestrs));
  }

  void
  ListForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    if (starts_ == Index::Form::i32) {
      builder.string("ListArray32");
    }
    else if (starts_ == Index::Form::u32) {
      builder.string("ListArrayU32");
    }
    else if (starts_ == Index::Form::i64) {
      builder.string("ListArray64");
    }
    else {
      builder.string("UnrecognizedListArray");
    }
    builder.field("starts");
    builder.string(Index::form2str(starts_));
    builder.field("stops");
    builder.string(Index::form2str(stops_));
    builder.field("content");
    content_.get()->tojson_part(builder, verbose);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  ListForm::shallow_copy() const {
    return std::make_shared<ListForm>(has_identities_,
                                      parameters_,
                                      form_key_,
                                      starts_,
                                      stops_,
                                      content_);
  }

  const std::string
  ListForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      return content_.get()->purelist_parameter(key);
    }
    else {
      return out;
    }
  }

  bool
  ListForm::purelist_isregular() const {
    return false;
  }

  int64_t
  ListForm::purelist_depth() const {
    return content_.get()->purelist_depth() + 1;
  }

  const std::pair<int64_t, int64_t>
  ListForm::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + 1,
                                       content_depth.second + 1);
  }

  const std::pair<bool, int64_t>
  ListForm::branch_depth() const {
    std::pair<bool, int64_t> content_depth = content_.get()->branch_depth();
    return std::pair<bool, int64_t>(content_depth.first,
                                    content_depth.second + 1);
  }

  int64_t
  ListForm::numfields() const {
    return content_.get()->numfields();
  }

  int64_t
  ListForm::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  const std::string
  ListForm::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  bool
  ListForm::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  const std::vector<std::string>
  ListForm::keys() const {
    return content_.get()->keys();
  }

  bool
  ListForm::equal(const FormPtr& other,
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
    if (ListForm* t = dynamic_cast<ListForm*>(other.get())) {
      return (starts_ == t->starts()  &&
              stops_ == t->stops()  &&
              content_.get()->equal(t->content(),
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
  ListForm::getitem_field(const std::string& key) const {
    return content_.get()->getitem_field(key);
  }

  ////////// ListArray

  template <typename T>
  ListArrayOf<T>::ListArrayOf(const IdentitiesPtr& identities,
                              const util::Parameters& parameters,
                              const IndexOf<T>& starts,
                              const IndexOf<T>& stops,
                              const ContentPtr& content)
      : Content(identities, parameters)
      , starts_(starts)
      , stops_(stops)
      , content_(content) {
    if (stops.length() < starts.length()) {
      throw std::invalid_argument(
        std::string("ListArray stops must not be shorter than its starts")
        + FILENAME(__LINE__));
    }
  }

  template <typename T>
  const IndexOf<T>
  ListArrayOf<T>::starts() const {
    return starts_;
  }

  template <typename T>
  const IndexOf<T>
  ListArrayOf<T>::stops() const {
    return stops_;
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::content() const {
    return content_;
  }

  template <typename T>
  Index64
  ListArrayOf<T>::compact_offsets64(bool start_at_zero) const {
    int64_t len = starts_.length();
    Index64 out(len + 1);
    struct Error err = kernel::ListArray_compact_offsets_64<T>(
      kernel::lib::cpu,   // DERIVE
      out.data(),
      starts_.data(),
      stops_.data(),
      len);
    util::handle_error(err, classname(), identities_.get());
    return out;
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::broadcast_tooffsets64(const Index64& offsets) const {
    if (offsets.length() == 0  ||  offsets.getitem_at_nowrap(0) != 0) {
      throw std::invalid_argument(
        std::string("broadcast_tooffsets64 can only be used with offsets that start at 0")
        + FILENAME(__LINE__));
    }
    if (offsets.length() - 1 > starts_.length()) {
      throw std::invalid_argument(
        std::string("cannot broadcast ListArray of length ")
        + std::to_string(starts_.length()) + (" to length ")
        + std::to_string(offsets.length() - 1) + FILENAME(__LINE__));
    }

    int64_t carrylen = offsets.getitem_at_nowrap(offsets.length() - 1);
    Index64 nextcarry(carrylen);
    struct Error err = kernel::ListArray_broadcast_tooffsets_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      offsets.data(),
      offsets.length(),
      starts_.data(),
      stops_.data(),
      content_.get()->length());
    util::handle_error(err, classname(), identities_.get());

    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);

    IdentitiesPtr identities;
    if (identities_.get() != nullptr) {
      identities =
        identities_.get()->getitem_range_nowrap(0, offsets.length() - 1);
    }
    return std::make_shared<ListOffsetArray64>(identities,
                                               parameters_,
                                               offsets,
                                               nextcontent);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::toRegularArray() const {
    Index64 offsets = compact_offsets64(true);
    ContentPtr listoffsetarray64 = broadcast_tooffsets64(offsets);
    ListOffsetArray64* raw =
      dynamic_cast<ListOffsetArray64*>(listoffsetarray64.get());
    return raw->toRegularArray();
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::toListOffsetArray64(bool start_at_zero) const {
    Index64 offsets = compact_offsets64(start_at_zero);
    return broadcast_tooffsets64(offsets);
  }

  template <typename T>
  const std::string
  ListArrayOf<T>::classname() const {
    if (std::is_same<T, int32_t>::value) {
      return "ListArray32";
    }
    else if (std::is_same<T, uint32_t>::value) {
      return "ListArrayU32";
    }
    else if (std::is_same<T, int64_t>::value) {
      return "ListArray64";
    }
    else {
      return "UnrecognizedListArray";
    }
  }

  template <typename T>
  void
  ListArrayOf<T>::setidentities(const IdentitiesPtr& identities) {
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
      if (content_.get()->length() > kMaxInt32  ||
          !std::is_same<T, int32_t>::value) {
        bigidentities = identities.get()->to64();
      }
      if (Identities32* rawidentities =
          dynamic_cast<Identities32*>(bigidentities.get())) {
        bool uniquecontents;
        IdentitiesPtr subidentities =
          std::make_shared<Identities32>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width() + 1,
                                         content_.get()->length());
        Identities32* rawsubidentities =
          reinterpret_cast<Identities32*>(subidentities.get());
        struct Error err = kernel::Identities_from_ListArray<int32_t, T>(
          kernel::lib::cpu,   // DERIVE
          &uniquecontents,
          rawsubidentities->data(),
          rawidentities->data(),
          starts_.data(),
          stops_.data(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else if (Identities64* rawidentities =
               dynamic_cast<Identities64*>(bigidentities.get())) {
        bool uniquecontents;
        IdentitiesPtr subidentities =
          std::make_shared<Identities64>(Identities::newref(),
                                         rawidentities->fieldloc(),
                                         rawidentities->width() + 1,
                                         content_.get()->length());
        Identities64* rawsubidentities =
          reinterpret_cast<Identities64*>(subidentities.get());
        struct Error err = kernel::Identities_from_ListArray<int64_t, T>(
          kernel::lib::cpu,   // DERIVE
          &uniquecontents,
          rawsubidentities->data(),
          rawidentities->data(),
          starts_.data(),
          stops_.data(),
          content_.get()->length(),
          length(),
          rawidentities->width());
        util::handle_error(err, classname(), identities_.get());
        if (uniquecontents) {
          content_.get()->setidentities(subidentities);
        }
        else {
          content_.get()->setidentities(Identities::none());
        }
      }
      else {
        throw std::runtime_error(
          std::string("unrecognized Identities specialization") + FILENAME(__LINE__));
      }
    }
    identities_ = identities;
  }

  template <typename T>
  void
  ListArrayOf<T>::setidentities() {
    if (length() <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1, length());
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_Identities<int32_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->ptr().get(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      IdentitiesPtr newidentities =
        std::make_shared<Identities64>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1, length());
      Identities64* rawidentities =
        reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = kernel::new_Identities<int64_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->ptr().get(),
        length());
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  template <typename T>
  const TypePtr
  ListArrayOf<T>::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  template <typename T>
  const FormPtr
  ListArrayOf<T>::form(bool materialize) const {
    return std::make_shared<ListForm>(identities_.get() != nullptr,
                                      parameters_,
                                      FormKey(nullptr),
                                      starts_.form(),
                                      stops_.form(),
                                      content_.get()->form(materialize));
  }

  template <typename T>
  bool
  ListArrayOf<T>::has_virtual_form() const {
    return content_.get()->has_virtual_form();
  }

  template <typename T>
  bool
  ListArrayOf<T>::has_virtual_length() const {
    return content_.get()->has_virtual_length();
  }

  template <typename T>
  const std::string
  ListArrayOf<T>::tostring_part(const std::string& indent,
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
    out << starts_.tostring_part(
             indent + std::string("    "), "<starts>", "</starts>\n");
    out << stops_.tostring_part(
             indent + std::string("    "), "<stops>", "</stops>\n");
    out << content_.get()->tostring_part(
             indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T>
  void
  ListArrayOf<T>::tojson_part(ToJson& builder,
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

  template <typename T>
  void
  ListArrayOf<T>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    starts_.nbytes_part(largest);
    stops_.nbytes_part(largest);
    content_.get()->nbytes_part(largest);
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  template <typename T>
  int64_t
  ListArrayOf<T>::length() const {
    return starts_.length();
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::shallow_copy() const {
    return std::make_shared<ListArrayOf<T>>(identities_,
                                            parameters_,
                                            starts_,
                                            stops_,
                                            content_);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::deep_copy(bool copyarrays,
                            bool copyindexes,
                            bool copyidentities) const {
    IndexOf<T> starts = copyindexes ? starts_.deep_copy() : starts_;
    IndexOf<T> stops = copyindexes ? stops_.deep_copy() : stops_;
    ContentPtr content = content_.get()->deep_copy(copyarrays,
                                                   copyindexes,
                                                   copyidentities);
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<ListArrayOf<T>>(identities,
                                            parameters_,
                                            starts,
                                            stops,
                                            content);
  }

  template <typename T>
  void
  ListArrayOf<T>::check_for_iteration() const {
    if (stops_.length() < starts_.length()) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < starts_.length()) {
      util::handle_error(
        failure("len(identities) < len(array)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_nothing() const {
    return content_.get()->getitem_range_nowrap(0, 0);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += starts_.length();
    }
    if (!(0 <= regular_at  &&  regular_at < starts_.length())) {
      util::handle_error(
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (regular_at >= stops_.length()) {
      util::handle_error(
        failure("len(stops) < len(starts)", kSliceNone, kSliceNone, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_at_nowrap(int64_t at) const {
    int64_t start = (int64_t)starts_.getitem_at_nowrap(at);
    int64_t stop = (int64_t)stops_.getitem_at_nowrap(at);
    int64_t lencontent = content_.get()->length();
    if (start == stop) {
      start = stop = 0;
    }
    if (start < 0) {
      util::handle_error(
        failure("starts[i] < 0", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (start > stop) {
      util::handle_error(
        failure("starts[i] > stops[i]", kSliceNone, at, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (stop > lencontent) {
      util::handle_error(
        failure("starts[i] != stops[i] and stops[i] > len(content)",
                kSliceNone,
                at,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return content_.get()->getitem_range_nowrap(start, stop);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), starts_.length());
    if (regular_stop > stops_.length()) {
      util::handle_error(
        failure("len(stops) < len(starts)", kSliceNone, kSliceNone, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range", kSliceNone, stop, FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<ListArrayOf<T>>(
      identities,
      parameters_,
      starts_.getitem_range_nowrap(start, stop),
      stops_.getitem_range_nowrap(start, stop),
      content_);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_field(const std::string& key) const {
    return std::make_shared<ListArrayOf<T>>(
      identities_,
      util::Parameters(),
      starts_,
      stops_,
      content_.get()->getitem_field(key));
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_fields(const std::vector<std::string>& keys) const {
    return std::make_shared<ListArrayOf<T>>(
      identities_,
      util::Parameters(),
      starts_,
      stops_,
      content_.get()->getitem_fields(keys));
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::carry(const Index64& carry, bool allow_lazy) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(
        failure("len(stops) < len(starts)", kSliceNone, kSliceNone, FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    IndexOf<T> nextstarts(carry.length());
    IndexOf<T> nextstops(carry.length());
    struct Error err = kernel::ListArray_getitem_carry_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextstarts.data(),
      nextstops.data(),
      starts_.data(),
      stops_.data(),
      carry.data(),
      lenstarts,
      carry.length());
    util::handle_error(err, classname(), identities_.get());
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<ListArrayOf<T>>(identities,
                                            parameters_,
                                            nextstarts,
                                            nextstops,
                                            content_);
  }

  template <typename T>
  int64_t
  ListArrayOf<T>::numfields() const {
    return content_.get()->numfields();
  }

  template <typename T>
  int64_t
  ListArrayOf<T>::fieldindex(const std::string& key) const {
    return content_.get()->fieldindex(key);
  }

  template <typename T>
  const std::string
  ListArrayOf<T>::key(int64_t fieldindex) const {
    return content_.get()->key(fieldindex);
  }

  template <typename T>
  bool
  ListArrayOf<T>::haskey(const std::string& key) const {
    return content_.get()->haskey(key);
  }

  template <typename T>
  const std::vector<std::string>
  ListArrayOf<T>::keys() const {
    return content_.get()->keys();
  }

  template <typename T>
  const std::string
  ListArrayOf<T>::validityerror(const std::string& path) const {
    if (stops_.length() < starts_.length()) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string("len(stops) < len(starts)")
              + FILENAME(__LINE__));
    }
    struct Error err = kernel::ListArray_validity<T>(
      kernel::lib::cpu,   // DERIVE
      starts_.data(),
      stops_.data(),
      starts_.length(),
      content_.get()->length());
    if (err.str == nullptr) {
      return content_.get()->validityerror(path + std::string(".content"));
    }
    else {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string(err.str)
              + std::string(" at i=") + std::to_string(err.identity)
              + std::string(err.filename == nullptr ? "" : err.filename));
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::shallow_simplify() const {
    return shallow_copy();
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else if (posaxis == depth + 1) {
      Index64 tonum(length());
      struct Error err = kernel::ListArray_num_64<T>(
        tonum.ptr_lib(),
        tonum.data(),
        starts_.data(),
        stops_.data(),
        length());
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<NumpyArray>(tonum);
    }
    else {
      return toListOffsetArray64(true).get()->num(posaxis, depth);
    }
  }

  template <typename T>
  const std::pair<Index64, ContentPtr>
  ListArrayOf<T>::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return toListOffsetArray64(true).get()->offsets_and_flattened(axis, depth);
  }

  template <typename T>
  bool
  ListArrayOf<T>::mergeable(const ContentPtr& other, bool mergebool) const {
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

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::merge(const ContentPtr& other) const {
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

    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index64 starts(mylength + theirlength);
    Index64 stops(mylength + theirlength);

    if (std::is_same<T, int32_t>::value) {
      struct Error err = kernel::ListArray_fill<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        0,
        stops.data(),
        0,
        reinterpret_cast<int32_t*>(starts_.data()),
        reinterpret_cast<int32_t*>(stops_.data()),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, uint32_t>::value) {
      struct Error err = kernel::ListArray_fill<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        0,
        stops.data(),
        0,
        reinterpret_cast<uint32_t*>(starts_.data()),
        reinterpret_cast<uint32_t*>(stops_.data()),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<T, int64_t>::value) {
      struct Error err = kernel::ListArray_fill<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        0,
        stops.data(),
        0,
        reinterpret_cast<int64_t*>(starts_.data()),
        reinterpret_cast<int64_t*>(stops_.data()),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized ListArray specialization") + FILENAME(__LINE__));
    }

    int64_t mycontentlength = content_.get()->length();
    ContentPtr content;
    if (ListArray32* rawother =
        dynamic_cast<ListArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_starts = rawother->starts();
      Index32 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (ListArrayU32* rawother =
             dynamic_cast<ListArrayU32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      IndexU32 other_starts = rawother->starts();
      IndexU32 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (ListArray64* rawother =
             dynamic_cast<ListArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (ListOffsetArray32* rawother =
             dynamic_cast<ListOffsetArray32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index32 other_starts = rawother->starts();
      Index32 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err, rawother->classname(),
                         rawother->identities().get());
    }
    else if (ListOffsetArrayU32* rawother =
             dynamic_cast<ListOffsetArrayU32*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      IndexU32 other_starts = rawother->starts();
      IndexU32 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (ListOffsetArray64* rawother =
             dynamic_cast<ListOffsetArray64*>(other.get())) {
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (RegularArray* rawregulararray =
             dynamic_cast<RegularArray*>(other.get())) {
      ContentPtr listoffsetarray = rawregulararray->toListOffsetArray64(true);
      ListOffsetArray64* rawother =
        dynamic_cast<ListOffsetArray64*>(listoffsetarray.get());
      content = content_.get()->merge(rawother->content());
      Index64 other_starts = rawother->starts();
      Index64 other_stops = rawother->stops();
      struct Error err = kernel::ListArray_fill<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        starts.data(),
        mylength,
        stops.data(),
        mylength,
        other_starts.data(),
        other_stops.data(),
        theirlength,
        mycontentlength);
      util::handle_error(err,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else {
      throw std::invalid_argument(
        std::string("cannot merge ") + classname() + std::string(" with ")
        + other.get()->classname() + FILENAME(__LINE__));
    }

    return std::make_shared<ListArray64>(Identities::none(),
                                         parameters_,
                                         starts,
                                         stops,
                                         content);
  }

  template <typename T>
  const SliceItemPtr
  ListArrayOf<T>::asslice() const {
    return toListOffsetArray64(true).get()->asslice();
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::fillna(const ContentPtr& value) const {
    return std::make_shared<ListArrayOf<T>>(identities_,
                                            parameters_,
                                            starts_,
                                            stops_,
                                            content_.get()->fillna(value));
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else if (posaxis == depth + 1) {
      int64_t min = target;
      struct Error err1 = kernel::ListArray_min_range<T>(
        kernel::lib::cpu,   // DERIVE
        &min,
        starts_.data(),
        stops_.data(),
        starts_.length());
      util::handle_error(err1, classname(), identities_.get());
      if (target < min) {
        return shallow_copy();
      }
      else {
        int64_t tolength = 0;
        struct Error err2 =
          kernel::ListArray_rpad_and_clip_length_axis1<T>(
          kernel::lib::cpu,   // DERIVE
          &tolength,
          starts_.data(),
          stops_.data(),
          target,
          starts_.length());
        util::handle_error(err2, classname(), identities_.get());

        Index64 index(tolength);
        IndexOf<T> starts(starts_.length());
        IndexOf<T> stops(starts_.length());
        struct Error err3 = kernel::ListArray_rpad_axis1_64<T>(
          kernel::lib::cpu,   // DERIVE
          index.data(),
          starts_.data(),
          stops_.data(),
          starts.data(),
          stops.data(),
          target,
          starts_.length());
        util::handle_error(err3, classname(), identities_.get());

        std::shared_ptr<IndexedOptionArray64> next =
          std::make_shared<IndexedOptionArray64>(Identities::none(),
                                                 util::Parameters(),
                                                 index, content());
        return std::make_shared<ListArrayOf<T>>(
          Identities::none(),
          parameters_,
          starts,
          stops,
          next.get()->simplify_optiontype());
      }
    }
    else {
      return std::make_shared<ListArrayOf<T>>(
        Identities::none(),
        parameters_,
        starts_,
        stops_,
        content_.get()->rpad(target, posaxis, depth + 1));
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::rpad_and_clip(int64_t target,
                                int64_t axis,
                                int64_t depth) const {
    return toListOffsetArray64(true).get()->rpad_and_clip(target, axis, depth);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::reduce_next(const Reducer& reducer,
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

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else if (posaxis == depth + 1) {
      Index64 offsets = compact_offsets64(true);
      int64_t innerlength =
        offsets.getitem_at_nowrap(offsets.length() - 1);
      Index64 localindex(innerlength);
      struct Error err = kernel::ListArray_localindex_64(
        kernel::lib::cpu,   // DERIVE
        localindex.data(),
        offsets.data(),
        offsets.length() - 1);
      util::handle_error(err, classname(), identities_.get());
      return std::make_shared<ListOffsetArray64>(
        identities_,
        util::Parameters(),
        offsets,
        std::make_shared<NumpyArray>(localindex));
    }
    else {
      return std::make_shared<ListArrayOf<T>>(
        identities_,
        util::Parameters(),
        starts_,
        stops_,
        content_.get()->localindex(posaxis, depth + 1));
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::combinations(int64_t n,
                               bool replacement,
                               const util::RecordLookupPtr& recordlookup,
                               const util::Parameters& parameters,
                               int64_t axis,
                               int64_t depth) const {
    if (n < 1) {
      throw std::invalid_argument(
        std::string("in combinations, 'n' must be at least 1") + FILENAME(__LINE__));
    }

    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return combinations_axis0(n, replacement, recordlookup, parameters);
    }

    else if (posaxis == depth + 1) {
      int64_t totallen;
      Index64 offsets(length() + 1);
      struct Error err1 = kernel::ListArray_combinations_length_64<T>(
        kernel::lib::cpu,   // DERIVE
        &totallen,
        offsets.data(),
        n,
        replacement,
        starts_.data(),
        stops_.data(),
        length());
      util::handle_error(err1, classname(), identities_.get());

      std::vector<std::shared_ptr<int64_t>> tocarry;
      std::vector<int64_t*> tocarryraw;
      for (int64_t j = 0;  j < n;  j++) {
        std::shared_ptr<int64_t> ptr(new int64_t[(size_t)totallen],
                                     kernel::array_deleter<int64_t>());
        tocarry.push_back(ptr);
        tocarryraw.push_back(ptr.get());
      }
      int64_t* toindex = new int64_t[n];
      int64_t* fromindex = new int64_t[n];
      struct Error err2 = kernel::ListArray_combinations_64<T>(
        kernel::lib::cpu,   // DERIVE
        tocarryraw.data(),
        toindex,
        fromindex,
        n,
        replacement,
        starts_.data(),
        stops_.data(),
        length());
      util::handle_error(err2, classname(), identities_.get());

      ContentPtrVec contents;
      for (auto ptr : tocarry) {
        contents.push_back(content_.get()->carry(
          Index64(ptr, 0, totallen, kernel::lib::cpu),   // DERIVE
        true));
      }
      ContentPtr recordarray = std::make_shared<RecordArray>(
        Identities::none(),
        parameters,
        contents,
        recordlookup);

      return std::make_shared<ListOffsetArray64>(identities_,
                                                 util::Parameters(),
                                                 offsets,
                                                 recordarray);
    }

    else {
      ContentPtr compact = toListOffsetArray64(true);
      ListOffsetArray64* rawcompact =
        dynamic_cast<ListOffsetArray64*>(compact.get());
      ContentPtr next = rawcompact->content().get()->combinations(n,
                                                                  replacement,
                                                                  recordlookup,
                                                                  parameters,
                                                                  posaxis,
                                                                  depth + 1);
      return std::make_shared<ListOffsetArray64>(identities_,
                                                 util::Parameters(),
                                                 rawcompact->offsets(),
                                                 next);
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::sort_next(int64_t negaxis,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength,
                            bool ascending,
                            bool stable,
                            bool keepdims) const {
    return toListOffsetArray64(true).get()->sort_next(negaxis,
                                                      starts,
                                                      parents,
                                                      outlength,
                                                      ascending,
                                                      stable,
                                                      keepdims);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::argsort_next(int64_t negaxis,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength,
                               bool ascending,
                               bool stable,
                               bool keepdims) const {
    return toListOffsetArray64(true).get()->argsort_next(negaxis,
                                                         starts,
                                                         parents,
                                                         outlength,
                                                         ascending,
                                                         stable,
                                                         keepdims);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next(const SliceAt& at,
                               const Slice& tail,
                               const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    if (advanced.length() != 0) {
      throw std::runtime_error(
        std::string("ListArray::getitem_next(SliceAt): advanced.length() != 0")
        + FILENAME(__LINE__));
    }
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 nextcarry(lenstarts);
    struct Error err = kernel::ListArray_getitem_next_at_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      starts_.data(),
      stops_.data(),
      lenstarts,
      at.at());
    util::handle_error(err, classname(), identities_.get());
    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);

    return nextcontent.get()->getitem_next(nexthead, nexttail, advanced);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next(const SliceRange& range,
                               const Slice& tail,
                               const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    int64_t start = range.start();
    int64_t stop = range.stop();
    int64_t step = range.step();
    if (step == Slice::none()) {
      step = 1;
    }
    int64_t carrylength;
    struct Error err1 =
      kernel::ListArray_getitem_next_range_carrylength<T>(
      kernel::lib::cpu,   // DERIVE
      &carrylength,
      starts_.data(),
      stops_.data(),
      lenstarts,
      start,
      stop,
      step);
    util::handle_error(err1, classname(), identities_.get());

    IndexOf<T> nextoffsets(lenstarts + 1);
    Index64 nextcarry(carrylength);

    struct Error err2 = kernel::ListArray_getitem_next_range_64<T>(
      kernel::lib::cpu,   // DERIVE
      nextoffsets.data(),
      nextcarry.data(),
      starts_.data(),
      stops_.data(),
      lenstarts,
      start,
      stop,
      step);
    util::handle_error(err2, classname(), identities_.get());
    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);

    if (advanced.length() == 0) {
      return std::make_shared<ListOffsetArrayOf<T>>(
        identities_,
        parameters_,
        nextoffsets,
        nextcontent.get()->getitem_next(nexthead, nexttail, advanced));
    }
    else {
      int64_t total;
      struct Error err1 = kernel::ListArray_getitem_next_range_counts_64<T>(
        kernel::lib::cpu,   // DERIVE
        &total,
        nextoffsets.data(),
        lenstarts);
      util::handle_error(err1, classname(), identities_.get());
      Index64 nextadvanced(total);
      struct Error err2 = kernel::ListArray_getitem_next_range_spreadadvanced_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextadvanced.data(),
        advanced.data(),
        nextoffsets.data(),
        lenstarts);
      util::handle_error(err2, classname(), identities_.get());
      return std::make_shared<ListOffsetArrayOf<T>>(
        identities_,
        parameters_,
        nextoffsets,
        nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced));
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next(const SliceArray64& array,
                               const Slice& tail,
                               const Index64& advanced) const {
    int64_t lenstarts = starts_.length();
    if (stops_.length() < lenstarts) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    Index64 flathead = array.ravel();
    if (advanced.length() == 0) {
      Index64 nextcarry(lenstarts*flathead.length());
      Index64 nextadvanced(lenstarts*flathead.length());
      struct Error err = kernel::ListArray_getitem_next_array_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        starts_.data(),
        stops_.data(),
        flathead.data(),
        lenstarts,
        flathead.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());
      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
      return getitem_next_array_wrap(
        nextcontent.get()->getitem_next(nexthead,
                                        nexttail,
                                        nextadvanced),
        array.shape());
    }
    else {
      Index64 nextcarry(lenstarts);
      Index64 nextadvanced(lenstarts);
      struct Error err = kernel::ListArray_getitem_next_array_advanced_64<T>(
        kernel::lib::cpu,   // DERIVE
        nextcarry.data(),
        nextadvanced.data(),
        starts_.data(),
        stops_.data(),
        flathead.data(),
        advanced.data(),
        lenstarts,
        flathead.length(),
        content_.get()->length());
      util::handle_error(err, classname(), identities_.get());
      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
      return nextcontent.get()->getitem_next(nexthead, nexttail, nextadvanced);
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next(const SliceJagged64& jagged,
                               const Slice& tail,
                               const Index64& advanced) const {
    if (advanced.length() != 0) {
      throw std::invalid_argument(
        std::string("cannot mix jagged slice with NumPy-style advanced indexing")
        + FILENAME(__LINE__));
    }
    if (stops_.length() < starts_.length()) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    int64_t len = length();
    Index64 singleoffsets = jagged.offsets();
    Index64 multistarts(jagged.length()*len);
    Index64 multistops(jagged.length()*len);
    Index64 nextcarry(jagged.length()*len);
    struct Error err = kernel::ListArray_getitem_jagged_expand_64(
      kernel::lib::cpu,   // DERIVE
      multistarts.data(),
      multistops.data(),
      singleoffsets.data(),
      nextcarry.data(),
      starts_.data(),
      stops_.data(),
      jagged.length(),
      len);
    util::handle_error(err, classname(), identities_.get());

    ContentPtr carried = content_.get()->carry(nextcarry, true);
    ContentPtr down = carried.get()->getitem_next_jagged(multistarts,
                                                         multistops,
                                                         jagged.content(),
                                                         tail);

    return std::make_shared<RegularArray>(Identities::none(),
                                          util::Parameters(),
                                          down, jagged.length());
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceArray64& slicecontent,
                                      const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(
        failure("jagged slice length differs from array length",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (stops_.length() < starts_.length()) {
      util::handle_error(
        failure("len(stops) < len(starts)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    int64_t carrylen;
    struct Error err1 = kernel::ListArray_getitem_jagged_carrylen_64(
      kernel::lib::cpu,   // DERIVE
      &carrylen,
      slicestarts.data(),
      slicestops.data(),
      slicestarts.length());
    util::handle_error(err1, classname(), identities_.get());

    Index64 sliceindex = slicecontent.index();
    Index64 outoffsets(slicestarts.length() + 1);
    Index64 nextcarry(carrylen);
    struct Error err2 = kernel::ListArray_getitem_jagged_apply_64<T>(
      kernel::lib::cpu,   // DERIVE
      outoffsets.data(),
      nextcarry.data(),
      slicestarts.data(),
      slicestops.data(),
      slicestarts.length(),
      sliceindex.data(),
      sliceindex.length(),
      starts_.data(),
      stops_.data(),
      content_.get()->length());
    util::handle_error(err2, classname(), nullptr);

    ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
    ContentPtr outcontent = nextcontent.get()->getitem_next(tail.head(),
                                                            tail.tail(),
                                                            Index64(0));

    return std::make_shared<ListOffsetArray64>(Identities::none(),
                                               util::Parameters(),
                                               outoffsets,
                                               outcontent);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceMissing64& slicecontent,
                                      const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(
        failure("jagged slice length differs from array length",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    Index64 missing = slicecontent.index();
    int64_t numvalid;
    struct Error err1 = kernel::ListArray_getitem_jagged_numvalid_64(
      kernel::lib::cpu,   // DERIVE
      &numvalid,
      slicestarts.data(),
      slicestops.data(),
      slicestarts.length(),
      missing.data(),
      missing.length());
    util::handle_error(err1, classname(), nullptr);

    Index64 nextcarry(numvalid);
    Index64 smalloffsets(slicestarts.length() + 1);
    Index64 largeoffsets(slicestarts.length() + 1);
    struct Error err2 = kernel::ListArray_getitem_jagged_shrink_64(
      kernel::lib::cpu,   // DERIVE
      nextcarry.data(),
      smalloffsets.data(),
      largeoffsets.data(),
      slicestarts.data(),
      slicestops.data(),
      slicestarts.length(),
      missing.data());
    util::handle_error(err2, classname(), nullptr);

    ContentPtr out;
    if (dynamic_cast<SliceJagged64*>(slicecontent.content().get())) {
      ContentPtr nextcontent = content_.get()->carry(nextcarry, true);
      ContentPtr next = std::make_shared<ListOffsetArray64>(Identities::none(),
                                                            util::Parameters(),
                                                            smalloffsets,
                                                            nextcontent);
      out = next.get()->getitem_next_jagged(util::make_starts(smalloffsets),
                                            util::make_stops(smalloffsets),
                                            slicecontent.content(),
                                            tail);
    }
    else {
      out = Content::getitem_next_jagged(util::make_starts(smalloffsets),
                                         util::make_stops(smalloffsets),
                                         slicecontent.content(),
                                         tail);
    }

    if (ListOffsetArray64* raw = dynamic_cast<ListOffsetArray64*>(out.get())) {
      ContentPtr content = raw->content();
      Index64 missing_trim =
          missing.getitem_range_nowrap(0, largeoffsets.getitem_at(-1));
      IndexedOptionArray64 indexedoptionarray(
          Identities::none(), util::Parameters(), missing_trim, content);
      return std::make_shared<ListOffsetArray64>(
        Identities::none(),
        util::Parameters(),
        largeoffsets,
        indexedoptionarray.simplify_optiontype());
    }
    else {
      throw std::runtime_error(
        std::string("expected ListOffsetArray64 from "
                    "ListArray::getitem_next_jagged, got ")
        + out.get()->classname() + FILENAME(__LINE__));
    }
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::getitem_next_jagged(const Index64& slicestarts,
                                      const Index64& slicestops,
                                      const SliceJagged64& slicecontent,
                                      const Slice& tail) const {
    if (starts_.length() < slicestarts.length()) {
      util::handle_error(
        failure("jagged slice length differs from array length",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }

    Index64 outoffsets(slicestarts.length() + 1);
    struct Error err = kernel::ListArray_getitem_jagged_descend_64<T>(
      kernel::lib::cpu,   // DERIVE
      outoffsets.data(),
      slicestarts.data(),
      slicestops.data(),
      slicestarts.length(),
      starts_.data(),
      stops_.data());
    util::handle_error(err, classname(), identities_.get());

    Index64 sliceoffsets = slicecontent.offsets();
    ContentPtr outcontent = content_.get()->getitem_next_jagged(
      util::make_starts(sliceoffsets),
      util::make_stops(sliceoffsets),
      slicecontent.content(),
      tail);

    return std::make_shared<ListOffsetArray64>(Identities::none(),
                                               util::Parameters(),
                                               outoffsets,
                                               outcontent);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::copy_to(kernel::lib ptr_lib) const {
    IndexOf<T> starts = starts_.copy_to(ptr_lib);
    IndexOf<T> stops = stops_.copy_to(ptr_lib);
    ContentPtr content= content_.get()->copy_to(ptr_lib);
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<ListArrayOf<T>>(identities,
                                            parameters_,
                                            starts,
                                            stops,
                                            content);
  }

  template <typename T>
  const ContentPtr
  ListArrayOf<T>::numbers_to_type(const std::string& name) const {
    IndexOf<T> starts = starts_.deep_copy();
    IndexOf<T> stops = stops_.deep_copy();
    ContentPtr content = content_.get()->numbers_to_type(name);
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<ListArrayOf<T>>(identities,
                                            parameters_,
                                            starts,
                                            stops,
                                            content);
  }

  template class EXPORT_TEMPLATE_INST ListArrayOf<int32_t>;
  template class EXPORT_TEMPLATE_INST ListArrayOf<uint32_t>;
  template class EXPORT_TEMPLATE_INST ListArrayOf<int64_t>;
}
