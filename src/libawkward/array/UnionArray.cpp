// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/UnionArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/UnionArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/type/UnionType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/type/UnknownType.h"
#include "awkward/Slice.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/VirtualArray.h"

#define AWKWARD_UNIONARRAY_NO_EXTERN_TEMPLATE
#include "awkward/array/UnionArray.h"

namespace awkward {
  ////////// UnionForm

  UnionForm::UnionForm(bool has_identities,
                       const util::Parameters& parameters,
                       const FormKey& form_key,
                       Index::Form tags,
                       Index::Form index,
                       const std::vector<FormPtr>& contents)
      : Form(has_identities, parameters, form_key)
      , tags_(tags)
      , index_(index)
      , contents_(contents) { }

  Index::Form
  UnionForm::tags() const {
    return tags_;
  }

  Index::Form
  UnionForm::index() const {
    return index_;
  }

  const std::vector<FormPtr>
  UnionForm::contents() const {
    return contents_;
  }

  int64_t
  UnionForm::numcontents() const {
    return (int64_t)contents_.size();
  }

  const FormPtr
  UnionForm::content(int64_t index) const {
    return contents_[(size_t)index];
  }

  const TypePtr
  UnionForm::type(const util::TypeStrs& typestrs) const {
    std::vector<TypePtr> types;
    for (auto item : contents_) {
      types.push_back(item.get()->type(typestrs));
    }
    return std::make_shared<UnionType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               types);
  }

  void
  UnionForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    if (index_ == Index::Form::i32) {
      builder.string("UnionArray8_32");
    }
    else if (index_ == Index::Form::u32) {
      builder.string("UnionArray8_U32");
    }
    else if (index_ == Index::Form::i64) {
      builder.string("UnionArray8_64");
    }
    else {
      builder.string("UnrecognizedUnionArray");
    }
    builder.field("tags");
    builder.string(Index::form2str(tags_));
    builder.field("index");
    builder.string(Index::form2str(index_));
    builder.field("contents");
    builder.beginlist();
    for (auto x : contents_) {
      x.get()->tojson_part(builder, verbose);
    }
    builder.endlist();
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  UnionForm::shallow_copy() const {
    return std::make_shared<UnionForm>(has_identities_,
                                       parameters_,
                                       form_key_,
                                       tags_,
                                       index_,
                                       contents_);
  }

  const std::string
  UnionForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      if (contents_.empty()) {
        return "null";
      }
      out = contents_[0].get()->purelist_parameter(key);
      for (size_t i = 1;  i < contents_.size();  i++) {
        if (!contents_[i].get()->parameter_equals(key, out)) {
          return "null";
        }
      }
      return out;
    }
    else {
      return out;
    }
  }

  bool
  UnionForm::purelist_isregular() const {
    for (auto content : contents_) {
      if (!content.get()->purelist_isregular()) {
        return false;
      }
    }
    return true;
  }

  int64_t
  UnionForm::purelist_depth() const {
    bool first = true;
    int64_t out = -1;
    for (auto content : contents_) {
      if (first) {
        first = false;
        out = content.get()->purelist_depth();
      }
      else if (out != content.get()->purelist_depth()) {
        return -1;
      }
    }
    return out;
  }

  const std::pair<int64_t, int64_t>
  UnionForm::minmax_depth() const {
    if (contents_.empty()) {
      return std::pair<int64_t, int64_t>(0, 0);
    }
    int64_t min = kMaxInt64;
    int64_t max = 0;
    for (auto content : contents_) {
      std::pair<int64_t, int64_t> minmax = content.get()->minmax_depth();
      if (minmax.first < min) {
        min = minmax.first;
      }
      if (minmax.second > max) {
        max = minmax.second;
      }
    }
    return std::pair<int64_t, int64_t>(min, max);
  }

  const std::pair<bool, int64_t>
  UnionForm::branch_depth() const {
    bool anybranch = false;
    int64_t mindepth = -1;
    for (auto content : contents_) {
      std::pair<bool, int64_t> content_depth = content.get()->branch_depth();
      if (mindepth == -1) {
        mindepth = content_depth.second;
      }
      if (content_depth.first  ||  mindepth != content_depth.second) {
        anybranch = true;
      }
      if (mindepth > content_depth.second) {
        mindepth = content_depth.second;
      }
    }
    return std::pair<bool, int64_t>(anybranch, mindepth);
  }

  int64_t
  UnionForm::numfields() const {
    return (int64_t)keys().size();
  }

  int64_t
  UnionForm::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys") + FILENAME(__LINE__));
  }

  const std::string
  UnionForm::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys") + FILENAME(__LINE__));
  }

  bool
  UnionForm::haskey(const std::string& key) const {
    for (auto x : keys()) {
      if (x == key) {
        return true;
      }
    }
    return false;
  }

  const std::vector<std::string>
  UnionForm::keys() const {
    std::vector<std::string> out;
    if (contents_.empty()) {
      return out;
    }
    out = contents_[0].get()->keys();
    for (size_t i = 1;  i < contents_.size();  i++) {
      std::vector<std::string> tmp = contents_[i].get()->keys();
      for (int64_t j = (int64_t)out.size() - 1;  j >= 0;  j--) {
        bool found = false;
        for (size_t k = 0;  k < tmp.size();  k++) {
          if (tmp[k] == out[(size_t)j]) {
            found = true;
            break;
          }
        }
        if (!found) {
          out.erase(std::next(out.begin(), j));
        }
      }
    }
    return out;
  }

  bool
  UnionForm::equal(const FormPtr& other,
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
    if (UnionForm* t = dynamic_cast<UnionForm*>(other.get())) {
      if (tags_ != t->tags()  ||  index_ != t->index()) {
        return false;
      }
      if (numcontents() != t->numcontents()) {
        return false;
      }
      for (int64_t i = 0;  i < numcontents();  i++) {
        if (!content(i).get()->equal(t->content(i),
                                     check_identities,
                                     check_parameters,
                                     check_form_key,
                                     compatibility_check)) {
          return false;
        }
      }
      return true;
    }
    else {
      return false;
    }
  }

  const FormPtr
  UnionForm::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      "UnionForm breaks the one-to-one relationship "
      "between fieldindexes and keys");
  }

  ////////// UnionArray

  template <>
  const IndexOf<int32_t>
  UnionArrayOf<int8_t, int32_t>::sparse_index(int64_t len) {
    IndexOf<int32_t> outindex(len);
    struct Error err = kernel::carry_arange<int32_t>(
      kernel::lib::cpu,   // DERIVE
      outindex.data(),
      len);
    util::handle_error(err, "UnionArray", nullptr);
    return outindex;
  }

  template <>
  const IndexOf<uint32_t>
  UnionArrayOf<int8_t, uint32_t>::sparse_index(int64_t len) {
    IndexOf<uint32_t> outindex(len);
    struct Error err = kernel::carry_arange<uint32_t>(
      kernel::lib::cpu,   // DERIVE
      outindex.data(),
      len);
    util::handle_error(err, "UnionArray", nullptr);
    return outindex;
  }

  template <>
  const IndexOf<int64_t>
  UnionArrayOf<int8_t, int64_t>::sparse_index(int64_t len) {
    IndexOf<int64_t> outindex(len);
    struct Error err = kernel::carry_arange<int64_t>(
      kernel::lib::cpu,   // DERIVE
      outindex.data(),
      len);
    util::handle_error(err, "UnionArray", nullptr);
    return outindex;
  }

  template <typename T, typename I>
  const IndexOf<I>
  UnionArrayOf<T, I>::regular_index(const IndexOf<T>& tags) {
    int64_t lentags = tags.length();
    int64_t size;
    struct Error err1 = kernel::UnionArray_regular_index_getsize<T>(
      kernel::lib::cpu,   // DERIVE
      &size,
      tags.data(),
      lentags);
    util::handle_error(err1, "UnionArray", nullptr);
    IndexOf<I> current(size);
    IndexOf<I> outindex(lentags);
    struct Error err2 = kernel::UnionArray_regular_index<T, I>(
      kernel::lib::cpu,   // DERIVE
      outindex.data(),
      current.data(),
      size,
      tags.data(),
      lentags);
    util::handle_error(err2, "UnionArray", nullptr);
    return outindex;
  }

  template <typename T, typename I>
  UnionArrayOf<T,
               I>::UnionArrayOf(const IdentitiesPtr& identities,
                                const util::Parameters& parameters,
                                const IndexOf<T> tags,
                                const IndexOf<I>& index,
                                const ContentPtrVec& contents)
      : Content(identities, parameters)
      , tags_(tags)
      , index_(index)
      , contents_(contents) {
    if (contents_.empty()) {
      throw std::invalid_argument("UnionArray must have at least one content");
    }
    if (index.length() < tags.length()) {
      throw std::invalid_argument(
        std::string("UnionArray index must not be shorter than its tags")
        + FILENAME(__LINE__));
    }
  }

  template <typename T, typename I>
  const IndexOf<T>
  UnionArrayOf<T, I>::tags() const {
    return tags_;
  }

  template <typename T, typename I>
  const IndexOf<I>
  UnionArrayOf<T, I>::index() const {
    return index_;
  }

  template <typename T, typename I>
  const ContentPtrVec
  UnionArrayOf<T, I>::contents() const {
    return contents_;
  }

  template <typename T, typename I>
  int64_t
  UnionArrayOf<T, I>::numcontents() const {
    return (int64_t)contents_.size();
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::content(int64_t index) const {
    if (!(0 <= index  &&  index < numcontents())) {
      throw std::invalid_argument(
        std::string("index ") + std::to_string(index)
        + std::string(" out of range for ") + classname()
        + std::string(" with ") + std::to_string(numcontents())
        + std::string(" contents") + FILENAME(__LINE__));
    }
    return contents_[(size_t)index];
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::project(int64_t index) const {
    if (!(0 <= index  &&  index < numcontents())) {
      throw std::invalid_argument(
        std::string("index ") + std::to_string(index)
        + std::string(" out of range for ") + classname()
        + std::string(" with ") + std::to_string(numcontents())
        + std::string(" contents") + FILENAME(__LINE__));
    }
    int64_t lentags = tags_.length();
    if (index_.length() < lentags) {
      util::handle_error(
        failure("len(index) < len(tags)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    int64_t lenout;
    Index64 tmpcarry(lentags);
    struct Error err = kernel::UnionArray_project_64<T, I>(
      kernel::lib::cpu,   // DERIVE
      &lenout,
      tmpcarry.data(),
      tags_.data(),
      index_.data(),
      lentags,
      index);
    util::handle_error(err, classname(), identities_.get());
    Index64 nextcarry(tmpcarry.ptr(), 0, lenout, tmpcarry.ptr_lib());
    return contents_[(size_t)index].get()->carry(nextcarry, false);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::simplify_uniontype(bool mergebool) const {
    int64_t len = length();
    if (index_.length() < len) {
      util::handle_error(
        failure("len(index) < len(tags)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    Index8 tags(len);
    Index64 index(len);
    ContentPtrVec contents;

    for (size_t i = 0;  i < contents_.size();  i++) {
      if (UnionArray8_32* rawcontent =
          dynamic_cast<UnionArray8_32*>(contents_[i].get())) {
        Index8 innertags = rawcontent->tags();
        Index32 innerindex = rawcontent->index();
        ContentPtrVec innercontents = rawcontent->contents();
        for (size_t j = 0;  j < innercontents.size();  j++) {
          bool unmerged = true;
          for (size_t k = 0;  k < contents.size();  k++) {
            if (contents[k].get()->mergeable(innercontents[j], mergebool)) {
              struct Error err = kernel::UnionArray_simplify8_32_to8_64<T, I>(
                kernel::lib::cpu,   // DERIVE
                tags.data(),
                index.data(),
                tags_.data(),
                index_.data(),
                innertags.data(),
                innerindex.data(),
                (int64_t)k,
                (int64_t)j,
                (int64_t)i,
                len,
                contents[k].get()->length());
              util::handle_error(err, classname(), identities_.get());
              contents[k] = contents[k].get()->merge(innercontents[j]);
              unmerged = false;
              break;
            }
          }
          if (unmerged) {
            struct Error err = kernel::UnionArray_simplify8_32_to8_64<T, I>(
              kernel::lib::cpu,   // DERIVE
              tags.data(),
              index.data(),
              tags_.data(),
              index_.data(),
              innertags.data(),
              innerindex.data(),
              (int64_t)contents.size(),
              (int64_t)j,
              (int64_t)i,
              len,
              0);
            util::handle_error(err, classname(), identities_.get());
            contents.push_back(innercontents[j]);
          }
        }
      }
      else if (UnionArray8_U32* rawcontent =
               dynamic_cast<UnionArray8_U32*>(contents_[i].get())) {
        Index8 innertags = rawcontent->tags();
        IndexU32 innerindex = rawcontent->index();
        ContentPtrVec innercontents = rawcontent->contents();
        for (size_t j = 0;  j < innercontents.size();  j++) {
          bool unmerged = true;
          for (size_t k = 0;  k < contents.size();  k++) {
            if (contents[k].get()->mergeable(innercontents[j], mergebool)) {
              struct Error err = kernel::UnionArray_simplify8_U32_to8_64<T, I>(
                kernel::lib::cpu,   // DERIVE
                tags.data(),
                index.data(),
                tags_.data(),
                index_.data(),
                innertags.data(),
                innerindex.data(),
                (int64_t)k,
                (int64_t)j,
                (int64_t)i,
                len,
                contents[k].get()->length());
              util::handle_error(err, classname(), identities_.get());
              contents[k] = contents[k].get()->merge(innercontents[j]);
              unmerged = false;
              break;
            }
          }
          if (unmerged) {
            struct Error err = kernel::UnionArray_simplify8_U32_to8_64<T, I>(
              kernel::lib::cpu,   // DERIVE
              tags.data(),
              index.data(),
              tags_.data(),
              index_.data(),
              innertags.data(),
              innerindex.data(),
              (int64_t)contents.size(),
              (int64_t)j,
              (int64_t)i,
              len,
              0);
            util::handle_error(err, classname(), identities_.get());
            contents.push_back(innercontents[j]);
          }
        }
      }
      else if (UnionArray8_64* rawcontent =
               dynamic_cast<UnionArray8_64*>(contents_[i].get())) {
        Index8 innertags = rawcontent->tags();
        Index64 innerindex = rawcontent->index();
        ContentPtrVec innercontents = rawcontent->contents();
        for (size_t j = 0;  j < innercontents.size();  j++) {
          bool unmerged = true;
          for (size_t k = 0;  k < contents.size();  k++) {
            if (contents[k].get()->mergeable(innercontents[j], mergebool)) {
              struct Error err = kernel::UnionArray_simplify8_64_to8_64<T, I>(
                kernel::lib::cpu,   // DERIVE
                tags.data(),
                index.data(),
                tags_.data(),
                index_.data(),
                innertags.data(),
                innerindex.data(),
                (int64_t)k,
                (int64_t)j,
                (int64_t)i,
                len,
                contents[k].get()->length());
              util::handle_error(err, classname(), identities_.get());
              contents[k] = contents[k].get()->merge(innercontents[j]);
              unmerged = false;
              break;
            }
          }
          if (unmerged) {
            struct Error err = kernel::UnionArray_simplify8_64_to8_64<T, I>(
              kernel::lib::cpu,   // DERIVE
              tags.data(),
              index.data(),
              tags_.data(),
              index_.data(),
              innertags.data(),
              innerindex.data(),
              (int64_t)contents.size(),
              (int64_t)j,
              (int64_t)i,
              len,
              0);
            util::handle_error(err, classname(), identities_.get());
            contents.push_back(innercontents[j]);
          }
        }
      }
      else {
        bool unmerged = true;
        for (size_t k = 0;  k < contents.size();  k++) {
          if (contents[k].get()->mergeable(contents_[i], mergebool)) {
            struct Error err = kernel::UnionArray_simplify_one_to8_64<T, I>(
              kernel::lib::cpu,   // DERIVE
              tags.data(),
              index.data(),
              tags_.data(),
              index_.data(),
              (int64_t)k,
              (int64_t)i,
              len,
              contents[k].get()->length());
            util::handle_error(err, classname(), identities_.get());
            contents[k] = contents[k].get()->merge(contents_[i]);
            unmerged = false;
            break;
          }
        }
        if (unmerged) {
          struct Error err = kernel::UnionArray_simplify_one_to8_64<T, I>(
            kernel::lib::cpu,   // DERIVE
            tags.data(),
            index.data(),
            tags_.data(),
            index_.data(),
            (int64_t)contents.size(),
            (int64_t)i,
            len,
            0);
          util::handle_error(err, classname(), identities_.get());
          contents.push_back(contents_[i]);
        }
      }
    }

    if (contents.size() > kMaxInt8) {
      throw std::runtime_error(
        std::string("FIXME: handle UnionArray with more than 127 contents")
        + FILENAME(__LINE__));
    }

    if (contents.size() == 1) {
      return contents[0].get()->carry(index, true);
    }
    else {
      return std::make_shared<UnionArray8_64>(identities_,
                                              parameters_,
                                              tags,
                                              index,
                                              contents);
    }
  }

  template <typename T, typename I>
  const std::string
  UnionArrayOf<T, I>::classname() const {
    if (std::is_same<T, int8_t>::value) {
      if (std::is_same<I, int32_t>::value) {
        return "UnionArray8_32";
      }
      else if (std::is_same<I, uint32_t>::value) {
        return "UnionArray8_U32";
      }
      else if (std::is_same<I, int64_t>::value) {
        return "UnionArray8_64";
      }
    }
    return "UnrecognizedUnionArray";
  }

  template <typename T, typename I>
  void
  UnionArrayOf<T, I>::setidentities() {
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

  template <typename T, typename I>
  void
  UnionArrayOf<T, I>::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() == nullptr) {
      for (auto content : contents_) {
        content.get()->setidentities(identities);
      }
    }
    else {
      if (index_.length() < tags_.length()) {
        util::handle_error(
          failure("len(index) < len(tags)",
                  kSliceNone,
                  kSliceNone,
                  FILENAME_C(__LINE__)),
          classname(),
          identities_.get());
      }
      if (length() != identities.get()->length()) {
        util::handle_error(
          failure("content and its identities must have the same length",
                  kSliceNone,
                  kSliceNone,
                  FILENAME_C(__LINE__)),
          classname(),
          identities_.get());
      }
      for (size_t which = 0;  which < contents_.size();  which++) {
        ContentPtr content = contents_[which];
        IdentitiesPtr bigidentities = identities;
        if (content.get()->length() > kMaxInt32  ||
            !std::is_same<I, int32_t>::value) {
          bigidentities = identities.get()->to64();
        }
        if (Identities32* rawidentities =
            dynamic_cast<Identities32*>(bigidentities.get())) {
          bool uniquecontents;
          IdentitiesPtr subidentities =
            std::make_shared<Identities32>(Identities::newref(),
                                           rawidentities->fieldloc(),
                                           rawidentities->width(),
                                           content.get()->length());
          Identities32* rawsubidentities =
            reinterpret_cast<Identities32*>(subidentities.get());
          struct Error err = kernel::Identities_from_UnionArray<int32_t, T, I>(
            kernel::lib::cpu,   // DERIVE
            &uniquecontents,
            rawsubidentities->data(),
            rawidentities->data(),
            tags_.data(),
            index_.data(),
            content.get()->length(),
            length(),
            rawidentities->width(),
            (int64_t)which);
          util::handle_error(err, classname(), identities_.get());
          if (uniquecontents) {
            content.get()->setidentities(subidentities);
          }
          else {
            content.get()->setidentities(Identities::none());
          }
        }
        else if (Identities64* rawidentities =
                 dynamic_cast<Identities64*>(bigidentities.get())) {
          bool uniquecontents;
          IdentitiesPtr subidentities =
            std::make_shared<Identities64>(Identities::newref(),
                                           rawidentities->fieldloc(),
                                           rawidentities->width(),
                                           content.get()->length());
          Identities64* rawsubidentities =
            reinterpret_cast<Identities64*>(subidentities.get());
          struct Error err = kernel::Identities_from_UnionArray<int64_t, T, I>(
            kernel::lib::cpu,   // DERIVE
            &uniquecontents,
            rawsubidentities->data(),
            rawidentities->data(),
            tags_.data(),
            index_.data(),
            content.get()->length(),
            length(),
            rawidentities->width(),
            (int64_t)which);
          util::handle_error(err, classname(), identities_.get());
          if (uniquecontents) {
            content.get()->setidentities(subidentities);
          }
          else {
            content.get()->setidentities(Identities::none());
          }
        }
        else {
          throw std::runtime_error(
            std::string("unrecognized Identities specialization")
            + FILENAME(__LINE__));
        }
      }
    }
    identities_ = identities;
  }

  template <typename T, typename I>
  const TypePtr
  UnionArrayOf<T, I>::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  template <typename T, typename I>
  const FormPtr
  UnionArrayOf<T, I>::form(bool materialize) const {
    std::vector<FormPtr> contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->form(materialize));
    }
    return std::make_shared<UnionForm>(identities_.get() != nullptr,
                                       parameters_,
                                       FormKey(nullptr),
                                       tags_.form(),
                                       index_.form(),
                                       contents);
  }

  template <typename T, typename I>
  bool
  UnionArrayOf<T, I>::has_virtual_form() const {
    for (auto x : contents_) {
      if (x.get()->has_virtual_form()) {
        return true;
      }
    }
    return false;
  }

  template <typename T, typename I>
  bool
  UnionArrayOf<T, I>::has_virtual_length() const {
    for (auto x : contents_) {
      if (x.get()->has_virtual_length()) {
        return true;
      }
    }
    return false;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayOf<T, I>::tostring_part(const std::string& indent,
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
    out << tags_.tostring_part(
             indent + std::string("    "), "<tags>", "</tags>\n");
    out << index_.tostring_part(
             indent + std::string("    "), "<index>", "</index>\n");
    for (size_t i = 0;  i < contents_.size();  i++) {
      out << indent << "    <content index=\"" << i << "\">\n";
      out << contents_[i].get()->tostring_part(
               indent + std::string("        "), "", "\n");
      out << indent << "    </content>\n";
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  template <typename T, typename I>
  void
  UnionArrayOf<T, I>::tojson_part(ToJson& builder,
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

  template <typename T, typename I>
  void
  UnionArrayOf<T, I>::nbytes_part(std::map<size_t, int64_t>& largest) const {
    for (auto x : contents_) {
      x.get()->nbytes_part(largest);
    }
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  template <typename T, typename I>
  int64_t
  UnionArrayOf<T, I>::length() const {
    return tags_.length();
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::shallow_copy() const {
    return std::make_shared<UnionArrayOf<T, I>>(identities_,
                                                parameters_,
                                                tags_,
                                                index_,
                                                contents_);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::deep_copy(bool copyarrays,
                                bool copyindexes,
                                bool copyidentities) const {
    IndexOf<T> tags = copyindexes ? tags_.deep_copy() : tags_;
    IndexOf<I> index = copyindexes ? index_.deep_copy() : index_;
    ContentPtrVec contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->deep_copy(copyarrays,
                                            copyindexes,
                                            copyidentities));
    }
    IdentitiesPtr identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities,
                                                parameters_,
                                                tags,
                                                index,
                                                contents);
  }

  template <typename T, typename I>
  void
  UnionArrayOf<T, I>::check_for_iteration() const {
    if (index_.length() < tags_.length()) {
      util::handle_error(
        failure("len(index) < len(tags)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    if (identities_.get() != nullptr  &&
        identities_.get()->length() < index_.length()) {
      util::handle_error(
        failure("len(identities) < len(array)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_nothing() const {
    return getitem_range_nowrap(0, 0);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_at(int64_t at) const {
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

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_at_nowrap(int64_t at) const {
    size_t tag = (size_t)tags_.getitem_at_nowrap(at);
    int64_t index = (int64_t)index_.getitem_at_nowrap(at);
    if (!(0 <= tag  &&  tag < contents_.size())) {
      util::handle_error(
        failure("not 0 <= tag[i] < numcontents",
                kSliceNone,
                at,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    ContentPtr content = contents_[tag];
    if (!(0 <= index  &&  index < content.get()->length())) {
      util::handle_error(
        failure("index[i] > len(content(tag))",
                kSliceNone,
                at,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return content.get()->getitem_at_nowrap(index);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), tags_.length());
    if (identities_.get() != nullptr  &&
        regular_stop > identities_.get()->length()) {
      util::handle_error(
        failure("index out of range", kSliceNone, stop, FILENAME_C(__LINE__)),
        identities_.get()->classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    return std::make_shared<UnionArrayOf<T, I>>(
      identities,
      parameters_,
      tags_.getitem_range_nowrap(start, stop),
      index_.getitem_range_nowrap(start, stop),
      contents_);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_field(const std::string& key) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->getitem_field(key));
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities_,
                                                util::Parameters(),
                                                tags_,
                                                index_,
                                                contents);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_fields(
    const std::vector<std::string>& keys) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->getitem_fields(keys));
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities_,
                                                util::Parameters(),
                                                tags_,
                                                index_,
                                                contents);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next(const SliceItemPtr& head,
                                   const Slice& tail,
                                   const Index64& advanced) const {
    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (dynamic_cast<SliceAt*>(head.get())  ||
             dynamic_cast<SliceRange*>(head.get())  ||
             dynamic_cast<SliceArray64*>(head.get())  ||
             dynamic_cast<SliceJagged64*>(head.get())) {
      ContentPtrVec outcontents;
      for (int64_t i = 0;  i < numcontents();  i++) {
        ContentPtr projection = project(i);
        outcontents.push_back(
          projection.get()->getitem_next(head, tail, advanced));
      }
      IndexOf<I> outindex = regular_index(tags_);
      UnionArrayOf<T, I> out(identities_,
                             parameters_,
                             tags_,
                             outindex,
                             outcontents);
      return out.simplify_uniontype(false);
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

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::carry(const Index64& carry, bool allow_lazy) const {
    int64_t lentags = tags_.length();
    if (index_.length() < lentags) {
      util::handle_error(
        failure("len(index) < len(tags)",
                kSliceNone,
                kSliceNone,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    int64_t lencarry = carry.length();
    IndexOf<T> nexttags(lencarry);
    struct Error err1 = kernel::Index_carry_64<T>(
      kernel::lib::cpu,   // DERIVE
      nexttags.data(),
      tags_.data(),
      carry.data(),
      lentags,
      lencarry);
    util::handle_error(err1, classname(), identities_.get());
    IndexOf<I> nextindex(lencarry);
    struct Error err2 = kernel::Index_carry_nocheck_64<I>(
      kernel::lib::cpu,   // DERIVE
      nextindex.data(),
      index_.data(),
      carry.data(),
      lencarry);
    util::handle_error(err2, classname(), identities_.get());
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities,
                                                parameters_,
                                                nexttags,
                                                nextindex,
                                                contents_);
  }

  template <typename T, typename I>
  int64_t
  UnionArrayOf<T, I>::numfields() const {
    return (int64_t)keys().size();
  }

  template <typename T, typename I>
  int64_t
  UnionArrayOf<T, I>::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  UnionArrayOf<T, I>::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys") + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  bool
  UnionArrayOf<T, I>::haskey(const std::string& key) const {
    for (auto x : keys()) {
      if (x == key) {
        return true;
      }
    }
    return false;
  }

  template <typename T, typename I>
  const std::vector<std::string>
  UnionArrayOf<T, I>::keys() const {
    std::vector<std::string> out;
    if (contents_.empty()) {
      return out;
    }
    out = contents_[0].get()->keys();
    for (size_t i = 1;  i < contents_.size();  i++) {
      std::vector<std::string> tmp = contents_[i].get()->keys();
      for (int64_t j = (int64_t)out.size() - 1;  j >= 0;  j--) {
        bool found = false;
        for (size_t k = 0;  k < tmp.size();  k++) {
          if (tmp[k] == out[(size_t)j]) {
            found = true;
            break;
          }
        }
        if (!found) {
          out.erase(std::next(out.begin(), j));
        }
      }
    }
    return out;
  }

  template <typename T, typename I>
  const std::string
  UnionArrayOf<T, I>::validityerror(const std::string& path) const {
    if (index_.length() < tags_.length()) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string("len(index) < len(tags)")
              + FILENAME(__LINE__));
    }
    std::vector<int64_t> lencontents;
    for (int64_t i = 0;  i < numcontents();  i++) {
      lencontents.push_back(content(i).get()->length());
    }
    struct Error err = kernel::UnionArray_validity<T, I>(
      kernel::lib::cpu,   // DERIVE
      tags_.data(),
      index_.data(),
      tags_.length(),
      numcontents(),
      lencontents.data());
    if (err.str != nullptr) {
      return (std::string("at ") + path + std::string(" (") + classname()
              + std::string("): ") + std::string(err.str)
              + std::string(" at i=") + std::to_string(err.identity)
              + std::string(err.filename == nullptr ? "" : err.filename));
    }
    for (int64_t i = 0;  i < numcontents();  i++) {
      std::string sub = content(i).get()->validityerror(
        path + std::string(".content(") + std::to_string(i) + (")"));
      if (!sub.empty()) {
        return sub;
      }
    }
    return std::string();
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::shallow_simplify() const {
    return simplify_uniontype(false);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 out(1);
      out.setitem_at_nowrap(0, length());
      return NumpyArray(out).getitem_at_nowrap(0);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->num(posaxis, depth));
      }
      UnionArrayOf<T, I> out(Identities::none(),
                             util::Parameters(),
                             tags_,
                             index_,
                             contents);
      return out.simplify_uniontype(false);
    }
  }

  template <typename T, typename I>
  const std::pair<Index64, ContentPtr>
  UnionArrayOf<T, I>::offsets_and_flattened(int64_t axis,
                                            int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else {
      bool has_offsets = false;
      std::vector<std::shared_ptr<int64_t>> offsetsptrs;
      std::vector<int64_t*> offsetsraws;
      ContentPtrVec contents;
      for (auto content : contents_) {
        std::pair<Index64, ContentPtr> pair =
          content.get()->offsets_and_flattened(posaxis, depth);
        Index64 offsets = pair.first;
        offsetsptrs.push_back(offsets.ptr());
        offsetsraws.push_back(offsets.data());
        contents.push_back(pair.second);
        has_offsets = (offsets.length() != 0);
      }

      if (has_offsets) {
        int64_t total_length;
        struct Error err1 = kernel::UnionArray_flatten_length_64<T, I>(
          kernel::lib::cpu,   // DERIVE
          &total_length,
          tags_.data(),
          index_.data(),
          tags_.length(),
          offsetsraws.data());
        util::handle_error(err1, classname(), identities_.get());

        Index8 totags(total_length);
        Index64 toindex(total_length);
        Index64 tooffsets(tags_.length() + 1);
        struct Error err2 = kernel::UnionArray_flatten_combine_64<T, I>(
          kernel::lib::cpu,   // DERIVE
          totags.data(),
          toindex.data(),
          tooffsets.data(),
          tags_.data(),
          index_.data(),
          tags_.length(),
          offsetsraws.data());
        util::handle_error(err2, classname(), identities_.get());
        return std::pair<Index64, ContentPtr>(
          tooffsets,
          std::make_shared<UnionArray8_64>(Identities::none(),
                                           util::Parameters(),
                                           totags,
                                           toindex,
                                           contents));
      }
      else {
        return std::pair<Index64, ContentPtr>(
          Index64(0),
          std::make_shared<UnionArrayOf<T, I>>(Identities::none(),
                                               util::Parameters(),
                                               tags_,
                                               index_,
                                               contents));
      }
    }
  }

  template <typename T, typename I>
  bool
  UnionArrayOf<T, I>::mergeable(const ContentPtr& other,
                                bool mergebool) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return mergeable(raw->array(), mergebool);
    }

    if (!parameters_equal(other.get()->parameters())) {
      return false;
    }
    return true;
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::reverse_merge(const ContentPtr& other) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return reverse_merge(raw->array());
    }

    int64_t theirlength = other.get()->length();
    int64_t mylength = length();
    Index8 tags(theirlength + mylength);
    Index64 index(theirlength + mylength);

    ContentPtrVec contents({ other });
    contents.insert(contents.end(), contents_.begin(), contents_.end());

    struct Error err1 = kernel::UnionArray_filltags_to8_const(
      kernel::lib::cpu,   // DERIVE
      tags.data(),
      0,
      theirlength,
      0);
    util::handle_error(err1, classname(), identities_.get());
    struct Error err2 = kernel::UnionArray_fillindex_count_64(
      kernel::lib::cpu,   // DERIVE
      index.data(),
      0,
      theirlength);
    util::handle_error(err2, classname(), identities_.get());

    if (std::is_same<T, int8_t>::value) {
      struct Error err = kernel::UnionArray_filltags_to8_from8(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        theirlength,
        reinterpret_cast<int8_t*>(tags_.data()),
        mylength,
        1);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized UnionArray specialization")
        + FILENAME(__LINE__));
    }

    if (std::is_same<I, int32_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<int32_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<I, uint32_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<uint32_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<I, int64_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        theirlength,
        reinterpret_cast<int64_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized UnionArray specialization")
        + FILENAME(__LINE__));
    }

    if (contents.size() > kMaxInt8) {
      throw std::runtime_error(
        std::string("FIXME: handle UnionArray with more than 127 contents")
        + FILENAME(__LINE__));
    }

    return std::make_shared<UnionArray8_64>(Identities::none(),
                                            parameters_,
                                            tags,
                                            index,
                                            contents);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::merge(const ContentPtr& other) const {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(other.get())) {
      return merge(raw->array());
    }

    if (!parameters_equal(other.get()->parameters())) {
      return merge_as_union(other);
    }

    if (dynamic_cast<EmptyArray*>(other.get())) {
      return shallow_copy();
    }

    int64_t mylength = length();
    int64_t theirlength = other.get()->length();
    Index8 tags(mylength + theirlength);
    Index64 index(mylength + theirlength);

    if (std::is_same<T, int8_t>::value) {
      struct Error err = kernel::UnionArray_filltags_to8_from8(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        0,
        reinterpret_cast<int8_t*>(tags_.data()),
        mylength,
        0);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized UnionArray specialization")
        + FILENAME(__LINE__));
    }

    if (std::is_same<I, int32_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        0,
        reinterpret_cast<int32_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<I, uint32_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        0,
        reinterpret_cast<uint32_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else if (std::is_same<I, int64_t>::value) {
      struct Error err = kernel::UnionArray_fillindex<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        0,
        reinterpret_cast<int64_t*>(index_.data()),
        mylength);
      util::handle_error(err, classname(), identities_.get());
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized UnionArray specialization")
        + FILENAME(__LINE__));
    }

    ContentPtrVec contents(contents_.begin(), contents_.end());
    if (UnionArray8_32* rawother =
        dynamic_cast<UnionArray8_32*>(other.get())) {
      ContentPtrVec other_contents = rawother->contents();
      contents.insert(contents.end(),
                      other_contents.begin(),
                      other_contents.end());
      Index8 other_tags = rawother->tags();
      struct Error err1 = kernel::UnionArray_filltags_to8_from8(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        mylength,
        other_tags.data(),
        theirlength,
        numcontents());
      util::handle_error(err1,
                         rawother->classname(),
                         rawother->identities().get());
      Index32 other_index = rawother->index();
      struct Error err2 = kernel::UnionArray_fillindex<int32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        mylength,
        other_index.data(),
        theirlength);
      util::handle_error(err2,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (UnionArray8_U32* rawother =
             dynamic_cast<UnionArray8_U32*>(other.get())) {
      ContentPtrVec other_contents = rawother->contents();
      contents.insert(contents.end(),
                      other_contents.begin(),
                      other_contents.end());
      Index8 other_tags = rawother->tags();
      struct Error err1 = kernel::UnionArray_filltags_to8_from8(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        mylength,
        other_tags.data(),
        theirlength,
        numcontents());
      util::handle_error(err1,
                         rawother->classname(),
                         rawother->identities().get());
      IndexU32 other_index = rawother->index();
      struct Error err2 = kernel::UnionArray_fillindex<uint32_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        mylength,
        other_index.data(),
        theirlength);
      util::handle_error(err2,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else if (UnionArray8_64* rawother =
             dynamic_cast<UnionArray8_64*>(other.get())) {
      ContentPtrVec other_contents = rawother->contents();
      contents.insert(contents.end(),
                      other_contents.begin(),
                      other_contents.end());
      Index8 other_tags = rawother->tags();
      struct Error err1 = kernel::UnionArray_filltags_to8_from8(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        mylength,
        other_tags.data(),
        theirlength,
        numcontents());
      util::handle_error(err1,
                         rawother->classname(),
                         rawother->identities().get());
      Index64 other_index = rawother->index();
      struct Error err2 = kernel::UnionArray_fillindex<int64_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        mylength,
        other_index.data(),
        theirlength);
      util::handle_error(err2,
                         rawother->classname(),
                         rawother->identities().get());
    }
    else {
      contents.push_back(other);
      struct Error err1 = kernel::UnionArray_filltags_to8_const(
        kernel::lib::cpu,   // DERIVE
        tags.data(),
        mylength,
        theirlength,
        numcontents());
      util::handle_error(err1, classname(), identities_.get());
      struct Error err2 = kernel::UnionArray_fillindex_count_64(
        kernel::lib::cpu,   // DERIVE
        index.data(),
        mylength,
        theirlength);
      util::handle_error(err2, classname(), identities_.get());
    }

    if (contents.size() > kMaxInt8) {
      throw std::runtime_error(
        std::string("FIXME: handle UnionArray with more than 127 contents")
        + FILENAME(__LINE__));
    }

    return std::make_shared<UnionArray8_64>(Identities::none(),
                                            parameters_,
                                            tags,
                                            index,
                                            contents);
  }

  template <typename T, typename I>
  const SliceItemPtr
  UnionArrayOf<T, I>::asslice() const {
    ContentPtr simplified = simplify_uniontype(false);
    if (UnionArray8_32* raw =
        dynamic_cast<UnionArray8_32*>(simplified.get())) {
      if (raw->numcontents() == 1) {
        return raw->content(0).get()->asslice();
      }
      else {
        throw std::invalid_argument(
          std::string("cannot use a union of different types as a slice")
          + FILENAME(__LINE__));
      }
    }
    else if (UnionArray8_U32* raw =
             dynamic_cast<UnionArray8_U32*>(simplified.get())) {
      if (raw->numcontents() == 1) {
        return raw->content(0).get()->asslice();
      }
      else {
        throw std::invalid_argument(
          std::string("cannot use a union of different types as a slice")
          + FILENAME(__LINE__));
      }
    }
    else if (UnionArray8_64* raw =
             dynamic_cast<UnionArray8_64*>(simplified.get())) {
      if (raw->numcontents() == 1) {
        return raw->content(0).get()->asslice();
      }
      else {
        throw std::invalid_argument(
          std::string("cannot use a union of different types as a slice")
          + FILENAME(__LINE__));
      }
    }
    else {
      return simplified.get()->asslice();
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::fillna(const ContentPtr& value) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.emplace_back(content.get()->fillna(value));
    }
    UnionArrayOf<T, I> out(identities_, parameters_, tags_, index_, contents);
    return out.simplify_uniontype(false);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.emplace_back(content.get()->rpad(target, posaxis, depth));
      }
      UnionArrayOf<T, I> out(identities_,
                             parameters_,
                             tags_,
                             index_,
                             contents);
      return out.simplify_uniontype(false);
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::rpad_and_clip(int64_t target,
                                    int64_t axis,
                                    int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, true);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.emplace_back(
          content.get()->rpad_and_clip(target, posaxis, depth));
      }
      UnionArrayOf<T, I> out(identities_,
                             parameters_,
                             tags_,
                             index_,
                             contents);
      return out.simplify_uniontype(false);
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::reduce_next(const Reducer& reducer,
                                  int64_t negaxis,
                                  const Index64& starts,
                                  const Index64& shifts,
                                  const Index64& parents,
                                  int64_t outlength,
                                  bool mask,
                                  bool keepdims) const {
    ContentPtr simplified = simplify_uniontype(true);
    if (dynamic_cast<UnionArray8_32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_U32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_64*>(simplified.get())) {
      throw std::invalid_argument(
        std::string("cannot reduce (call '") + reducer.name()
        + std::string("' on) an irreducible ") + classname()
        + FILENAME(__LINE__));
    }
    return simplified.get()->reduce_next(reducer,
                                         negaxis,
                                         starts,
                                         shifts,
                                         parents,
                                         outlength,
                                         mask,
                                         keepdims);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->localindex(posaxis, depth));
      }
      return std::make_shared<UnionArrayOf<T, I>>(identities_,
                                                  util::Parameters(),
                                                  tags_,
                                                  index_,
                                                  contents);
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::combinations(int64_t n,
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
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->combinations(n,
                                                       replacement,
                                                       recordlookup,
                                                       parameters,
                                                       posaxis,
                                                       depth));
      }
      return std::make_shared<UnionArrayOf<T, I>>(identities_,
                                                  util::Parameters(),
                                                  tags_,
                                                  index_,
                                                  contents);
    }
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::sort_next(int64_t negaxis,
                                const Index64& starts,
                                const Index64& parents,
                                int64_t outlength,
                                bool ascending,
                                bool stable,
                                bool keepdims) const {
    ContentPtr simplified = simplify_uniontype(true);
    if (dynamic_cast<UnionArray8_32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_U32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_64*>(simplified.get())) {
      throw std::invalid_argument(
        std::string("cannot sort ") + classname() + FILENAME(__LINE__));
    }
    return simplified.get()->sort_next(negaxis,
                                       starts,
                                       parents,
                                       outlength,
                                       ascending,
                                       stable,
                                       keepdims);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::argsort_next(int64_t negaxis,
                                   const Index64& starts,
                                   const Index64& parents,
                                   int64_t outlength,
                                   bool ascending,
                                   bool stable,
                                   bool keepdims) const {
    ContentPtr simplified = simplify_uniontype(true);
    if (dynamic_cast<UnionArray8_32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_U32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_64*>(simplified.get())) {
      throw std::invalid_argument(
        std::string("cannot sort ") + classname() + FILENAME(__LINE__));
    }
    return simplified.get()->argsort_next(negaxis,
                                          starts,
                                          parents,
                                          outlength,
                                          ascending,
                                          stable,
                                          keepdims);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next(const SliceAt& at,
                                   const Slice& tail,
                                   const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnionArray::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next(const SliceRange& range,
                                   const Slice& tail,
                                   const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnionArray::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next(const SliceArray64& array,
                                   const Slice& tail,
                                   const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnionArray::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next(const SliceJagged64& jagged,
                                   const Slice& tail,
                                   const Index64& advanced) const {
    throw std::runtime_error(
      std::string("undefined operation: UnionArray::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next_jagged(const Index64& slicestarts,
                                          const Index64& slicestops,
                                          const SliceArray64& slicecontent,
                                          const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts,
                                                     slicestops,
                                                     slicecontent,
                                                     tail);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next_jagged(const Index64& slicestarts,
                                          const Index64& slicestops,
                                          const SliceMissing64& slicecontent,
                                          const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next_jagged(const Index64& slicestarts,
                                          const Index64& slicestops,
                                          const SliceJagged64& slicecontent,
                                          const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts,
                                                      slicestops,
                                                      slicecontent,
                                                      tail);
  }


  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::copy_to(kernel::lib ptr_lib) const {
    IndexOf<T> tags = tags_.copy_to(ptr_lib);
    IndexOf<I> index = index_.copy_to(ptr_lib);
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->copy_to(ptr_lib));
    }
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities,
                                                parameters_,
                                                tags,
                                                index,
                                                contents);
  }

  template <typename T, typename I>
  const ContentPtr
  UnionArrayOf<T, I>::numbers_to_type(const std::string& name) const {
    IndexOf<T> tags = tags_.deep_copy();
    IndexOf<I> index = index_.deep_copy();
    ContentPtrVec contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->numbers_to_type(name));
    }
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<UnionArrayOf<T, I>>(identities,
                                                parameters_,
                                                tags,
                                                index,
                                                contents);
  }

  template <typename T, typename I>
  template <typename S>
  const ContentPtr
  UnionArrayOf<T, I>::getitem_next_jagged_generic(const Index64& slicestarts,
                                                  const Index64& slicestops,
                                                  const S& slicecontent,
                                                  const Slice& tail) const {
    ContentPtr simplified = simplify_uniontype(false);
    if (dynamic_cast<UnionArray8_32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_U32*>(simplified.get())  ||
        dynamic_cast<UnionArray8_64*>(simplified.get())) {
      throw std::invalid_argument(
        std::string("cannot apply jagged slices to irreducible union arrays")
        + FILENAME(__LINE__));
    }
    return simplified.get()->getitem_next_jagged(slicestarts,
                                                 slicestops,
                                                 slicecontent,
                                                 tail);
  }

  template class EXPORT_TEMPLATE_INST UnionArrayOf<int8_t, int32_t>;
  template class EXPORT_TEMPLATE_INST UnionArrayOf<int8_t, uint32_t>;
  template class EXPORT_TEMPLATE_INST UnionArrayOf<int8_t, int64_t>;
}
