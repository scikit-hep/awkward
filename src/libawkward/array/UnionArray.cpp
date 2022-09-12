// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/UnionArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/UnionArray.cpp", line)

#include <sstream>
#include <type_traits>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"



#include "awkward/io/json.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/NumpyArray.h"


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

  const FormPtr
  UnionForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<UnionForm>(has_identities_,
                                       parameters_,
                                       form_key,
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
        if (!util::json_equals(out, contents_[i].get()->purelist_parameter(key))) {
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

  bool
  UnionForm::dimension_optiontype() const {
    for (auto content : contents_) {
      if (content.get()->dimension_optiontype()) {
        return true;
      }
    }
    return false;
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
  UnionForm::istuple() const {
    bool all_contents_are_tuple = true;
    for (auto content : contents_) {
        all_contents_are_tuple = all_contents_are_tuple && content.get()->istuple();
    }
    return all_contents_are_tuple && (!contents_.empty());
  }

  bool
  UnionForm::equal(const FormPtr& other,
                   bool check_identities,
                   bool check_parameters,
                   bool check_form_key,
                   bool compatibility_check) const {
    if (compatibility_check) {

    }

    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters(), false)) {
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
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys")
      + FILENAME(__LINE__));
  }

  const FormPtr
  UnionForm::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("UnionForm breaks the one-to-one relationship "
                  "between fieldindexes and keys")
      + FILENAME(__LINE__));
  }
}
