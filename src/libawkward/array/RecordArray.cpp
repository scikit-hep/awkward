// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/RecordArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/RecordArray.cpp", line)

#include <sstream>
#include <algorithm>

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"


#include "awkward/io/json.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/RegularArray.h"


#include "awkward/array/RecordArray.h"

namespace awkward {
  ////////// RecordForm

  RecordForm::RecordForm(bool has_identities,
                         const util::Parameters& parameters,
                         const FormKey& form_key,
                         const util::RecordLookupPtr& recordlookup,
                         const std::vector<FormPtr>& contents)
      : Form(has_identities, parameters, form_key)
      , recordlookup_(recordlookup)
      , contents_(contents) {
    if (recordlookup.get() != nullptr  &&
        recordlookup.get()->size() != contents.size()) {
      throw std::invalid_argument(
        std::string("recordlookup (if provided) and contents must have the same length")
        + FILENAME(__LINE__));
    }
  }

  const util::RecordLookupPtr
  RecordForm::recordlookup() const {
    return recordlookup_;
  }

  const std::vector<FormPtr>
  RecordForm::contents() const {
    return contents_;
  }

  bool
  RecordForm::istuple() const {
    return recordlookup_.get() == nullptr;
  }

  const FormPtr
  RecordForm::content(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(
        std::string("fieldindex ") + std::to_string(fieldindex)
        + std::string(" for record with only ") + std::to_string(numfields())
        + std::string(" fields") + FILENAME(__LINE__));
    }
    return contents_[(size_t)fieldindex];
  }

  const FormPtr
  RecordForm::content(const std::string& key) const {
    return contents_[(size_t)fieldindex(key)];
  }

  const std::vector<std::pair<std::string, FormPtr>>
  RecordForm::items() const {
    std::vector<std::pair<std::string, FormPtr>> out;
    if (recordlookup_.get() != nullptr) {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, FormPtr>(recordlookup_.get()->at(j),
                                          contents_[j]));
      }
    }
    else {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, FormPtr>(std::to_string(j),
                                          contents_[j]));
      }
    }
    return out;
  }

  void
  RecordForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("RecordArray");
    builder.field("contents");
    if (recordlookup_.get() == nullptr) {
      builder.beginlist();
      for (auto x : contents_) {
        x.get()->tojson_part(builder, verbose);
      }
      builder.endlist();
    }
    else {
      builder.beginrecord();
      for (size_t i = 0;  i < recordlookup_.get()->size();  i++) {
        builder.field(recordlookup_.get()->at(i));
        contents_[i].get()->tojson_part(builder, verbose);
      }
      builder.endrecord();
    }
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    form_key_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  RecordForm::shallow_copy() const {
    return std::make_shared<RecordForm>(has_identities_,
                                        parameters_,
                                        form_key_,
                                        recordlookup_,
                                        contents_);
  }

  const FormPtr
  RecordForm::with_form_key(const FormKey& form_key) const {
    return std::make_shared<RecordForm>(has_identities_,
                                        parameters_,
                                        form_key,
                                        recordlookup_,
                                        contents_);
  }

  const std::string
  RecordForm::purelist_parameter(const std::string& key) const {
    return parameter(key);
  }

  bool
  RecordForm::purelist_isregular() const {
    return true;
  }

  int64_t
  RecordForm::purelist_depth() const {
    return 1;
  }

  bool
  RecordForm::dimension_optiontype() const {
    return false;
  }

  const std::pair<int64_t, int64_t>
  RecordForm::minmax_depth() const {
    if (contents_.empty()) {
      return std::pair<int64_t, int64_t>(1, 1);
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
  RecordForm::branch_depth() const {
    if (contents_.empty()) {
      return std::pair<bool, int64_t>(false, 1);
    }
    else {
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
  }

  int64_t
  RecordForm::numfields() const {
    return (int64_t)contents_.size();
  }

  int64_t
  RecordForm::fieldindex(const std::string& key) const {
    return util::fieldindex(recordlookup_, key, numfields());
  }

  const std::string
  RecordForm::key(int64_t fieldindex) const {
    return util::key(recordlookup_, fieldindex, numfields());
  }

  bool
  RecordForm::haskey(const std::string& key) const {
    return util::haskey(recordlookup_, key, numfields());
  }

  const std::vector<std::string>
  RecordForm::keys() const {
    return util::keys(recordlookup_, numfields());
  }

  bool
  RecordForm::equal(const FormPtr& other,
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
    if (RecordForm* t = dynamic_cast<RecordForm*>(other.get())) {
      if (recordlookup_.get() == nullptr  &&
          t->recordlookup().get() != nullptr) {
        return false;
      }
      else if (recordlookup_.get() != nullptr  &&
               t->recordlookup().get() == nullptr) {
        return false;
      }
      else if (recordlookup_.get() != nullptr  &&
               t->recordlookup().get() != nullptr) {
        util::RecordLookupPtr one = recordlookup_;
        util::RecordLookupPtr two = t->recordlookup();
        if (one.get()->size() != two.get()->size()) {
          return false;
        }
        for (int64_t i = 0;  i < (int64_t)one.get()->size();  i++) {
          int64_t j = 0;
          for (;  j < (int64_t)one.get()->size();  j++) {
            if (one.get()->at((uint64_t)i) == two.get()->at((uint64_t)j)) {
              break;
            }
          }
          if (j == (int64_t)one.get()->size()) {
            return false;
          }
          if (!content(i).get()->equal(t->content(j),
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
        if (numfields() != t->numfields()) {
          return false;
        }
        for (int64_t i = 0;  i < numfields();  i++) {
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
    }
    else {
      return false;
    }
  }

  const FormPtr
  RecordForm::getitem_field(const std::string& key) const {
    return content(key);
  }

  const FormPtr
  RecordForm::getitem_fields(const std::vector<std::string>& keys) const {
    util::RecordLookupPtr recordlookup(nullptr);
    if (recordlookup_.get() != nullptr) {
      recordlookup = std::make_shared<util::RecordLookup>();
    }
    std::vector<FormPtr> contents;
    for (auto key : keys) {
      if (recordlookup_.get() != nullptr) {
        recordlookup.get()->push_back(key);
      }
      contents.push_back(contents_[(size_t)fieldindex(key)]);
    }
    return std::make_shared<RecordForm>(has_identities_,
                                        util::Parameters(),
                                        FormKey(nullptr),
                                        recordlookup,
                                        contents);
  }
}
