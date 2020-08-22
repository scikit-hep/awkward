// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/array/RecordArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/array/RecordArray.cpp", line)

#include <sstream>
#include <algorithm>

#include "awkward/kernels/identities.h"
#include "awkward/kernels/getitem.h"
#include "awkward/kernels/operations.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/Record.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/VirtualArray.h"

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

  const TypePtr
  RecordForm::type(const util::TypeStrs& typestrs) const {
    std::vector<TypePtr> types;
    for (auto item : contents_) {
      types.push_back(item.get()->type(typestrs));
    }
    return std::make_shared<RecordType>(
               parameters_,
               util::gettypestr(parameters_, typestrs),
               types,
               recordlookup_);
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

  const std::pair<int64_t, int64_t>
  RecordForm::minmax_depth() const {
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

  ////////// RecordArray

  RecordArray::RecordArray(const IdentitiesPtr& identities,
                           const util::Parameters& parameters,
                           const ContentPtrVec& contents,
                           const util::RecordLookupPtr& recordlookup,
                           int64_t length)
      : Content(identities, parameters)
      , contents_(contents)
      , recordlookup_(recordlookup)
      , length_(length) {
    if (recordlookup_.get() != nullptr  &&
        recordlookup_.get()->size() != contents_.size()) {
      throw std::invalid_argument(
        std::string("recordlookup and contents must have the same number of fields")
        + FILENAME(__LINE__));
    }
  }

  int64_t minlength(const ContentPtrVec& contents) {
    if (contents.empty()) {
      return 0;
    }
    else {
      int64_t out = -1;
      for (auto x : contents) {
        int64_t len = x.get()->length();
        if (out < 0  ||  out > len) {
          out = len;
        }
      }
      return out;
    }
  }

  RecordArray::RecordArray(const IdentitiesPtr& identities,
                           const util::Parameters& parameters,
                           const ContentPtrVec& contents,
                           const util::RecordLookupPtr& recordlookup)
      : RecordArray(identities,
                    parameters,
                    contents,
                    recordlookup,
                    minlength(contents)) { }

  const ContentPtrVec
  RecordArray::contents() const {
    return contents_;
  }

  const util::RecordLookupPtr
  RecordArray::recordlookup() const {
    return recordlookup_;
  }

  bool
  RecordArray::istuple() const {
    return recordlookup_.get() == nullptr;
  }

  const ContentPtr
  RecordArray::setitem_field(int64_t where, const ContentPtr& what) const {
    if (where < 0) {
      throw std::invalid_argument(
        std::string("where must be non-negative") + FILENAME(__LINE__));
    }
    if (what.get()->length() != length()) {
      throw std::invalid_argument(
        std::string("array of length ") + std::to_string(what.get()->length())
        + std::string(" cannot be assigned to record array of length ")
        + std::to_string(length()) + FILENAME(__LINE__));
    }
    ContentPtrVec contents;
    for (size_t i = 0;  i < contents_.size();  i++) {
      if (where == (int64_t)i) {
        contents.push_back(what);
      }
      contents.push_back(contents_[i]);
    }
    if (where >= numfields()) {
      contents.push_back(what);
    }
    util::RecordLookupPtr recordlookup(nullptr);
    if (recordlookup_.get() != nullptr) {
      recordlookup = std::make_shared<util::RecordLookup>();
      for (size_t i = 0;  i < contents_.size();  i++) {
        if (where == (int64_t)i) {
          recordlookup.get()->push_back(std::to_string(where));
        }
        recordlookup.get()->push_back(recordlookup_.get()->at(i));
      }
      if (where >= numfields()) {
        recordlookup.get()->push_back(std::to_string(where));
      }
    }
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents,
                                         recordlookup);
  }

  const ContentPtr
  RecordArray::setitem_field(const std::string& where,
                             const ContentPtr& what) const {
    if (what.get()->length() != length()) {
      throw std::invalid_argument(
        std::string("array of length ") + std::to_string(what.get()->length())
        + std::string(" cannot be assigned to record array of length ")
        + std::to_string(length()) + FILENAME(__LINE__));
    }
    ContentPtrVec contents(contents_.begin(), contents_.end());
    contents.push_back(what);
    util::RecordLookupPtr recordlookup;
    if (recordlookup_.get() != nullptr) {
      recordlookup = std::make_shared<util::RecordLookup>();
      recordlookup.get()->insert(recordlookup.get()->end(),
                                 recordlookup_.get()->begin(),
                                 recordlookup_.get()->end());
      recordlookup.get()->push_back(where);
    }
    else {
      recordlookup = util::init_recordlookup(numfields());
      recordlookup.get()->push_back(where);
    }
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents,
                                         recordlookup);
  }

  const std::string
  RecordArray::classname() const {
    return "RecordArray";
  }

  void
  RecordArray::setidentities() {
    int64_t len = length();
    if (len <= kMaxInt32) {
      IdentitiesPtr newidentities =
        std::make_shared<Identities32>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1,
                                       len);
      Identities32* rawidentities =
        reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = kernel::new_Identities<int32_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        len);
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      IdentitiesPtr newidentities =
        std::make_shared<Identities64>(Identities::newref(),
                                       Identities::FieldLoc(),
                                       1, len);
      Identities64* rawidentities =
        reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = kernel::new_Identities<int64_t>(
        kernel::lib::cpu,   // DERIVE
        rawidentities->data(),
        len);
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  void
  RecordArray::setidentities(const IdentitiesPtr& identities) {
    if (identities.get() == nullptr) {
      for (auto content : contents_) {
        content.get()->setidentities(identities);
      }
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
      if (istuple()) {
        Identities::FieldLoc original = identities.get()->fieldloc();
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identities::FieldLoc fieldloc(original.begin(), original.end());
          fieldloc.push_back(
            std::pair<int64_t, std::string>(identities.get()->width() - 1,
                                            std::to_string(j)));
          contents_[j].get()->setidentities(
            identities.get()->withfieldloc(fieldloc));
        }
      }
      else {
        Identities::FieldLoc original = identities.get()->fieldloc();
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identities::FieldLoc fieldloc(original.begin(), original.end());
          fieldloc.push_back(std::pair<int64_t, std::string>(
            identities.get()->width() - 1, recordlookup_.get()->at(j)));
          contents_[j].get()->setidentities(
            identities.get()->withfieldloc(fieldloc));
        }
      }
    }
    identities_ = identities;
  }

  const TypePtr
  RecordArray::type(const util::TypeStrs& typestrs) const {
    return form(true).get()->type(typestrs);
  }

  const FormPtr
  RecordArray::form(bool materialize) const {
    std::vector<FormPtr> contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->form(materialize));
    }
    return std::make_shared<RecordForm>(identities_.get() != nullptr,
                                        parameters_,
                                        FormKey(nullptr),
                                        recordlookup_,
                                        contents);
  }

  bool
  RecordArray::has_virtual_form() const {
    for (auto x : contents_) {
      if (x.get()->has_virtual_form()) {
        return true;
      }
    }
    return false;
  }

  bool
  RecordArray::has_virtual_length() const {
    for (auto x : contents_) {
      if (x.get()->has_virtual_length()) {
        return true;
      }
    }
    return false;
  }

  const std::string
  RecordArray::tostring_part(const std::string& indent,
                             const std::string& pre,
                             const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (contents_.empty()) {
      out << " length=\"" << length_ << "\"";
    }
    out << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    for (size_t j = 0;  j < contents_.size();  j++) {
      out << indent << "    <field index=\"" << j << "\"";
      if (!istuple()) {
        out << " key=\"" << recordlookup_.get()->at(j) << "\">";
      }
      else {
        out << ">";
      }
      out << "\n";
      out << contents_[j].get()->tostring_part(
               indent + std::string("        "), "", "\n");
      out << indent << "    </field>\n";
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  RecordArray::tojson_part(ToJson& builder, bool include_beginendlist) const {
    int64_t rows = length();
    size_t cols = contents_.size();
    util::RecordLookupPtr keys = recordlookup_;
    if (istuple()) {
      keys = std::make_shared<util::RecordLookup>();
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    check_for_iteration();
    if (include_beginendlist) {
      builder.beginlist();
    }
    for (int64_t i = 0;  i < rows;  i++) {
      builder.beginrecord();
      for (size_t j = 0;  j < cols;  j++) {
        builder.field(keys.get()->at(j).c_str());
        contents_[j].get()->getitem_at_nowrap(i).get()->tojson_part(builder,
                                                                    true);
      }
      builder.endrecord();
    }
    if (include_beginendlist) {
      builder.endlist();
    }
  }

  void
  RecordArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    for (auto x : contents_) {
      x.get()->nbytes_part(largest);
    }
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t
  RecordArray::length() const {
    return length_;
  }

  const ContentPtr
  RecordArray::shallow_copy() const {
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents_,
                                         recordlookup_,
                                         length_);
  }

  const ContentPtr
  RecordArray::deep_copy(bool copyarrays,
                         bool copyindexes,
                         bool copyidentities) const {
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
    return std::make_shared<RecordArray>(identities,
                                         parameters_,
                                         contents,
                                         recordlookup_,
                                         length_);
  }

  void
  RecordArray::check_for_iteration() const {
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
  RecordArray::getitem_nothing() const {
    return getitem_range_nowrap(0, 0);
  }

  const ContentPtr
  RecordArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    int64_t len = length();
    if (regular_at < 0) {
      regular_at += len;
    }
    if (!(0 <= regular_at  &&  regular_at < len)) {
      util::handle_error(
        failure("index out of range",
                kSliceNone,
                at,
                FILENAME_C(__LINE__)),
        classname(),
        identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  RecordArray::getitem_at_nowrap(int64_t at) const {
    return std::make_shared<Record>(shared_from_this(), at);
  }

  const ContentPtr
  RecordArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length_);
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
  RecordArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_range_nowrap(start, stop);
    }
    if (contents_.empty()) {
      return std::make_shared<RecordArray>(identities,
                                           parameters_,
                                           contents_,
                                           recordlookup_,
                                           stop - start);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_range_nowrap(start, stop));
      }
      return std::make_shared<RecordArray>(identities,
                                           parameters_,
                                           contents,
                                           recordlookup_,
                                           stop - start);
    }
  }

  const ContentPtr
  RecordArray::getitem_field(const std::string& key) const {
    return field(key).get()->getitem_range_nowrap(0, length());
  }

  const ContentPtr
  RecordArray::getitem_fields(const std::vector<std::string>& keys) const {
    ContentPtrVec contents;
    util::RecordLookupPtr recordlookup(nullptr);
    if (recordlookup_.get() != nullptr) {
      recordlookup = std::make_shared<util::RecordLookup>();
    }
    for (auto key : keys) {
      contents.push_back(field(key).get()->getitem_range_nowrap(0, length()));
      if (recordlookup.get() != nullptr) {
        recordlookup.get()->push_back(key);
      }
    }
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents,
                                         recordlookup);
  }

  const ContentPtr
  RecordArray::carry(const Index64& carry, bool allow_lazy) const {
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->getitem_carry_64(carry);
    }
    if (allow_lazy) {
      return std::make_shared<IndexedArray64>(identities,
                                              parameters_,
                                              carry,
                                              shallow_copy());
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->carry(carry, allow_lazy));
      }
      return std::make_shared<RecordArray>(identities,
                                           parameters_,
                                           contents,
                                           recordlookup_,
                                           carry.length());
    }
  }

  int64_t
  RecordArray::numfields() const {
    return (int64_t)contents_.size();
  }

  int64_t
  RecordArray::fieldindex(const std::string& key) const {
    return util::fieldindex(recordlookup_, key, numfields());
  }

  const std::string
  RecordArray::key(int64_t fieldindex) const {
    return util::key(recordlookup_, fieldindex, numfields());
  }

  bool
  RecordArray::haskey(const std::string& key) const {
    return util::haskey(recordlookup_, key, numfields());
  }

  const std::vector<std::string>
  RecordArray::keys() const {
    return util::keys(recordlookup_, numfields());
  }

  const std::string
  RecordArray::validityerror(const std::string& path) const {
    for (int64_t i = 0;  i < numfields();  i++) {
      if (field(i).get()->length() < length_) {
        return (std::string("at ") + path + std::string(" (") + classname()
                + std::string("): len(field(")
                + std::to_string(i) + (")) < len(recordarray)")
                + FILENAME(__LINE__));
      }
    }
    for (int64_t i = 0;  i < numfields();  i++) {
      std::string sub = field(i).get()->validityerror(
        path + std::string(".field(") + std::to_string(i) + (")"));
      if (!sub.empty()) {
        return sub;
      }
    }
    return std::string();
  }

  const ContentPtr
  RecordArray::shallow_simplify() const {
    return shallow_copy();
  }

  const ContentPtr
  RecordArray::num(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      Index64 single(1);
      single.setitem_at_nowrap(0, length_);
      ContentPtr singleton = std::make_shared<NumpyArray>(single);
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(singleton);
      }
      ContentPtr record = std::make_shared<RecordArray>(Identities::none(),
                                                        util::Parameters(),
                                                        contents,
                                                        recordlookup_,
                                                        1);
      return record.get()->getitem_at_nowrap(0);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->num(posaxis, depth));
      }
      return std::make_shared<RecordArray>(Identities::none(),
                                           util::Parameters(),
                                           contents,
                                           recordlookup_,
                                           length_);
    }
  }

  const std::pair<Index64, ContentPtr>
  RecordArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      throw std::invalid_argument(
        std::string("axis=0 not allowed for flatten") + FILENAME(__LINE__));
    }
    else if (posaxis == depth + 1) {
      throw std::invalid_argument(
        std::string("arrays of records cannot be flattened (but their contents can be; "
                    "try a different 'axis')") + FILENAME(__LINE__));
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        ContentPtr trimmed = content.get()->getitem_range(0, length());
        std::pair<Index64, ContentPtr> pair =
          trimmed.get()->offsets_and_flattened(posaxis, depth);
        if (pair.first.length() != 0) {
          throw std::runtime_error(
            std::string("RecordArray content with axis > depth + 1 returned a non-empty "
                        "offsets from offsets_and_flattened") + FILENAME(__LINE__));
        }
        contents.push_back(pair.second);
      }
      return std::pair<Index64, ContentPtr>(
        Index64(0),
        std::make_shared<RecordArray>(Identities::none(),
                                      util::Parameters(),
                                      contents,
                                      recordlookup_));
    }
  }

  bool
  RecordArray::mergeable(const ContentPtr& other, bool mergebool) const {
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

    if (RecordArray* rawother =
        dynamic_cast<RecordArray*>(other.get())) {
      if (istuple()  &&  rawother->istuple()) {
        if (numfields() == rawother->numfields()) {
          for (int64_t i = 0;  i < numfields();  i++) {
            if (!field(i).get()->mergeable(rawother->field(i), mergebool)) {
              return false;
            }
          }
          return true;
        }
      }
      else if (!istuple()  &&  !rawother->istuple()) {
        std::vector<std::string> self_keys = keys();
        std::vector<std::string> other_keys = rawother->keys();
        std::sort(self_keys.begin(), self_keys.end());
        std::sort(other_keys.begin(), other_keys.end());
        if (self_keys == other_keys) {
          for (auto key : self_keys) {
            if (!field(key).get()->mergeable(rawother->field(key),
                                             mergebool)) {
              return false;
            }
          }
          return true;
        }
      }
      return false;
    }
    else {
      return false;
    }
  }

  const ContentPtr
  RecordArray::merge(const ContentPtr& other) const {
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

    if (RecordArray* rawother =
        dynamic_cast<RecordArray*>(other.get())) {
      int64_t mylength = length();
      int64_t theirlength = rawother->length();

      if (istuple() == rawother->istuple()  &&
          numfields() == 0  &&  rawother->numfields() == 0) {
        return std::make_shared<RecordArray>(Identities::none(),
                                             parameters_,
                                             contents_,
                                             util::RecordLookupPtr(nullptr),
                                             mylength + theirlength);
      }
      if (istuple()  &&  rawother->istuple()) {
        if (numfields() == rawother->numfields()) {
          ContentPtrVec contents;
          for (int64_t i = 0;  i < numfields();  i++) {
            ContentPtr mine =
              field(i).get()->getitem_range_nowrap(0, mylength);
            ContentPtr theirs =
              rawother->field(i).get()->getitem_range_nowrap(0, theirlength);
            contents.push_back(mine.get()->merge(theirs));
          }
          return std::make_shared<RecordArray>(Identities::none(),
                                               parameters_,
                                               contents,
                                               recordlookup_);
        }
      }
      else if (!istuple()  &&  !rawother->istuple()) {
        std::vector<std::string> self_keys = keys();
        std::vector<std::string> other_keys = rawother->keys();
        std::sort(self_keys.begin(), self_keys.end());
        std::sort(other_keys.begin(), other_keys.end());
        if (self_keys == other_keys) {
          ContentPtrVec contents;
          for (auto key : keys()) {
            ContentPtr mine =
              field(key).get()->getitem_range_nowrap(0, mylength);
            ContentPtr theirs =
              rawother->field(key).get()->getitem_range_nowrap(0, theirlength);
            contents.push_back(mine.get()->merge(theirs));
          }
          return std::make_shared<RecordArray>(Identities::none(),
                                               parameters_,
                                               contents,
                                               recordlookup_);
        }
      }
      throw std::invalid_argument(
        std::string("cannot merge records or tuples with different fields")
        + FILENAME(__LINE__));
    }
    else {
      throw std::invalid_argument(
        std::string("cannot merge ") + classname() + std::string(" with ")
        + other.get()->classname() + FILENAME(__LINE__));
    }
  }

  const SliceItemPtr
  RecordArray::asslice() const {
    throw std::invalid_argument(
      std::string("cannot use records as a slice") + FILENAME(__LINE__));
  }

  const ContentPtr
  RecordArray::fillna(const ContentPtr& value) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->fillna(value));
    }
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents,
                                         recordlookup_,
                                         length_);
  }

  const ContentPtr
  RecordArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, false);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->rpad(target, posaxis, depth));
      }
      if (contents.empty()) {
        return std::make_shared<RecordArray>(identities_,
                                             parameters_,
                                             contents,
                                             recordlookup_,
                                             length_);
      }
      else {
        return std::make_shared<RecordArray>(identities_,
                                             parameters_,
                                             contents,
                                             recordlookup_);
      }
    }
  }

  const ContentPtr
  RecordArray::rpad_and_clip(int64_t target,
                             int64_t axis,
                             int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return rpad_axis0(target, true);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(
          content.get()->rpad_and_clip(target, posaxis, depth));
      }
      if (contents.empty()) {
        return std::make_shared<RecordArray>(identities_,
                                             parameters_,
                                             contents,
                                             recordlookup_,
                                             length_);
      }
      else {
        return std::make_shared<RecordArray>(identities_,
                                             parameters_,
                                             contents,
                                             recordlookup_);
      }
    }
  }

  const ContentPtr
  RecordArray::reduce_next(const Reducer& reducer,
                           int64_t negaxis,
                           const Index64& starts,
                           const Index64& shifts,
                           const Index64& parents,
                           int64_t outlength,
                           bool mask,
                           bool keepdims) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      ContentPtr trimmed = content.get()->getitem_range_nowrap(0, length());
      ContentPtr next = trimmed.get()->reduce_next(reducer,
                                                   negaxis,
                                                   starts,
                                                   shifts,
                                                   parents,
                                                   outlength,
                                                   mask,
                                                   keepdims);
      contents.push_back(next);
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         util::Parameters(),
                                         contents,
                                         recordlookup_,
                                         outlength);
  }

  const ContentPtr
  RecordArray::localindex(int64_t axis, int64_t depth) const {
    int64_t posaxis = axis_wrap_if_negative(axis);
    if (posaxis == depth) {
      return localindex_axis0();
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->localindex(posaxis, depth));
      }
      return std::make_shared<RecordArray>(identities_,
                                           util::Parameters(),
                                           contents,
                                           recordlookup_,
                                           length_);
    }
  }

  const ContentPtr
  RecordArray::combinations(int64_t n,
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
      return std::make_shared<RecordArray>(identities_,
                                           util::Parameters(),
                                           contents,
                                           recordlookup_,
                                           length_);
    }
  }

  const ContentPtr
  RecordArray::field(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(
        std::string("fieldindex ") + std::to_string(fieldindex)
        + std::string(" for record with only " + std::to_string(numfields()))
        + std::string(" fields") + FILENAME(__LINE__));
    }
    return contents_[(size_t)fieldindex];
  }

  const ContentPtr
  RecordArray::field(const std::string& key) const {
    return contents_[(size_t)fieldindex(key)];
  }

  const ContentPtrVec
  RecordArray::fields() const {
    return ContentPtrVec(contents_);
  }

  const std::vector<std::pair<std::string, ContentPtr>>
  RecordArray::fielditems() const {
    std::vector<std::pair<std::string, ContentPtr>> out;
    if (istuple()) {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, ContentPtr>(std::to_string(j), contents_[j]));
      }
    }
    else {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(
          std::pair<std::string, ContentPtr>(recordlookup_.get()->at(j),
                                             contents_[j]));
      }
    }
    return out;
  }

  const std::shared_ptr<RecordArray>
  RecordArray::astuple() const {
    return std::make_shared<RecordArray>(identities_,
                                         parameters_,
                                         contents_,
                                         util::RecordLookupPtr(nullptr),
                                         length_);
  }

  const ContentPtr
  RecordArray::sort_next(int64_t negaxis,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength,
                         bool ascending,
                         bool stable,
                         bool keepdims) const {
    std::vector<ContentPtr> contents;
    for (auto content : contents_) {
      ContentPtr trimmed = content.get()->getitem_range_nowrap(0, length());
      ContentPtr next = trimmed.get()->sort_next(negaxis,
                                                 starts,
                                                 parents,
                                                 outlength,
                                                 ascending,
                                                 stable,
                                                 keepdims);
      contents.push_back(next);
    }
    return std::make_shared<RecordArray>(Identities::none(),
                                         parameters_,
                                         contents,
                                         recordlookup_,
                                         outlength);
  }

  const ContentPtr
  RecordArray::argsort_next(int64_t negaxis,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength,
                            bool ascending,
                            bool stable,
                            bool keepdims) const {
    std::vector<ContentPtr> contents;
    for (auto content : contents_) {
      ContentPtr trimmed = content.get()->getitem_range_nowrap(0, length());
      ContentPtr next = trimmed.get()->argsort_next(negaxis,
                                                    starts,
                                                    parents,
                                                    outlength,
                                                    ascending,
                                                    stable,
                                                    keepdims);
      contents.push_back(next);
    }
    return std::make_shared<RecordArray>(
      Identities::none(),
      util::Parameters(),
      contents,
      recordlookup_,
      outlength);
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceItemPtr& head,
                            const Slice& tail,
                            const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    Slice emptytail;
    emptytail.become_sealed();

    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceField* field =
             dynamic_cast<SliceField*>(head.get())) {
      ContentPtr out = getitem_next(*field, emptytail, advanced);
      return out.get()->getitem_next(nexthead, nexttail, advanced);
    }
    else if (SliceFields* fields =
             dynamic_cast<SliceFields*>(head.get())) {
      ContentPtr out = getitem_next(*fields, emptytail, advanced);
      return out.get()->getitem_next(nexthead, nexttail, advanced);
    }
    else if (const SliceMissing64* missing =
             dynamic_cast<SliceMissing64*>(head.get())) {
      return Content::getitem_next(*missing, tail, advanced);
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_next(head,
                                                       emptytail,
                                                       advanced));
      }
      util::Parameters parameters;
      if (head.get()->preserves_type(advanced)) {
        parameters = parameters_;
      }
      RecordArray out(Identities::none(), parameters, contents, recordlookup_);
      return out.getitem_next(nexthead, nexttail, advanced);
    }
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceAt& at,
                            const Slice& tail,
                            const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("undefined operation: RecordArray::getitem_next(at)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceRange& range,
                            const Slice& tail,
                            const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("undefined operation: RecordArray::getitem_next(range)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceArray64& array,
                            const Slice& tail,
                            const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("undefined operation: RecordArray::getitem_next(array)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceField& field,
                            const Slice& tail,
                            const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_field(field.key()).get()->getitem_next(nexthead,
                                                          nexttail,
                                                          advanced);
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceFields& fields,
                            const Slice& tail,
                            const Index64& advanced) const {
    SliceItemPtr nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_fields(fields.keys()).get()->getitem_next(nexthead,
                                                             nexttail,
                                                             advanced);
  }

  const ContentPtr
  RecordArray::getitem_next(const SliceJagged64& jagged,
                            const Slice& tail,
                            const Index64& advanced) const {
    throw std::invalid_argument(
      std::string("undefined operation: RecordArray::getitem_next(jagged)")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  RecordArray::getitem_next_jagged(const Index64& slicestarts,
                                   const Index64& slicestops,
                                   const SliceArray64& slicecontent,
                                   const Slice& tail) const {
    return getitem_next_jagged_generic<SliceArray64>(slicestarts,
                                                     slicestops,
                                                     slicecontent,
                                                     tail);
  }

  const ContentPtr
  RecordArray::getitem_next_jagged(const Index64& slicestarts,
                                   const Index64& slicestops,
                                   const SliceMissing64& slicecontent,
                                   const Slice& tail) const {
    return getitem_next_jagged_generic<SliceMissing64>(slicestarts,
                                                       slicestops,
                                                       slicecontent,
                                                       tail);
  }

  const ContentPtr
  RecordArray::getitem_next_jagged(const Index64& slicestarts,
                                   const Index64& slicestops,
                                   const SliceJagged64& slicecontent,
                                   const Slice& tail) const {
    return getitem_next_jagged_generic<SliceJagged64>(slicestarts,
                                                      slicestops,
                                                      slicecontent,
                                                      tail);
  }

  const ContentPtr
  RecordArray::copy_to(kernel::lib ptr_lib) const {
    ContentPtrVec contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->copy_to(ptr_lib));
    }
    IdentitiesPtr identities(nullptr);
    if (identities_.get() != nullptr) {
      identities = identities_.get()->copy_to(ptr_lib);
    }
    return std::make_shared<RecordArray>(identities,
                                         parameters_,
                                         contents,
                                         recordlookup_,
                                         length_);
  }

  const ContentPtr
  RecordArray::numbers_to_type(const std::string& name) const {
    ContentPtrVec contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->numbers_to_type(name));
    }
    IdentitiesPtr identities = identities_;
    if (identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    return std::make_shared<RecordArray>(identities,
                                         parameters_,
                                         contents,
                                         recordlookup_,
                                         length_);
  }

  template <typename S>
  const ContentPtr
  RecordArray::getitem_next_jagged_generic(const Index64& slicestarts,
                                           const Index64& slicestops,
                                           const S& slicecontent,
                                           const Slice& tail) const {
    if (contents_.empty()) {
      return shallow_copy();
    }
    else {
      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_next_jagged(slicestarts,
                                                              slicestops,
                                                              slicecontent,
                                                              tail));
      }
      return std::make_shared<RecordArray>(identities_,
                                           parameters_,
                                           contents,
                                           recordlookup_);
    }
  }

}
