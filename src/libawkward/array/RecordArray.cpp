// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identities.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/RecordType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/Record.h"

#include "awkward/array/RecordArray.h"

namespace awkward {
  RecordArray::RecordArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::vector<std::shared_ptr<Content>>& contents, const std::shared_ptr<util::RecordLookup>& recordlookup)
      : Content(identities, parameters)
      , contents_(contents)
      , recordlookup_(recordlookup)
      , length_(0) {
    if (contents_.empty()) {
      throw std::runtime_error("this constructor can only be used with non-empty contents");
    }
    if (recordlookup_.get() != nullptr  &&  recordlookup_.get()->size() != contents_.size()) {
      throw std::runtime_error("recordlookup and contents must have the same length");
    }
  }

  RecordArray::RecordArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const std::vector<std::shared_ptr<Content>>& contents)
      : Content(identities, parameters)
      , contents_(contents)
      , recordlookup_(nullptr)
      , length_(0) {
    if (contents_.empty()) {
      throw std::runtime_error("this constructor can only be used with non-empty contents");
    }
  }

  RecordArray::RecordArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, int64_t length, bool istuple)
      : Content(identities, parameters)
      , contents_()
      , recordlookup_(istuple ? nullptr : new util::RecordLookup)
      , length_(length) { }

  const std::vector<std::shared_ptr<Content>> RecordArray::contents() const {
    return contents_;
  }

  const std::shared_ptr<util::RecordLookup> RecordArray::recordlookup() const {
    return recordlookup_;
  }

  bool RecordArray::istuple() const {
    return recordlookup_.get() == nullptr;
  }

  const std::string RecordArray::classname() const {
    return "RecordArray";
  }

  void RecordArray::setidentities() {
    int64_t len = length();
    if (len <= kMaxInt32) {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities32>(Identities::newref(), Identities::FieldLoc(), 1, len);
      Identities32* rawidentities = reinterpret_cast<Identities32*>(newidentities.get());
      struct Error err = awkward_new_identities32(rawidentities->ptr().get(), len);
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
    else {
      std::shared_ptr<Identities> newidentities = std::make_shared<Identities64>(Identities::newref(), Identities::FieldLoc(), 1, len);
      Identities64* rawidentities = reinterpret_cast<Identities64*>(newidentities.get());
      struct Error err = awkward_new_identities64(rawidentities->ptr().get(), len);
      util::handle_error(err, classname(), identities_.get());
      setidentities(newidentities);
    }
  }

  void RecordArray::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() == nullptr) {
      for (auto content : contents_) {
        content.get()->setidentities(identities);
      }
    }
    else {
      if (length() != identities.get()->length()) {
        util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
      }
      if (istuple()) {
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identities::FieldLoc fieldloc(identities.get()->fieldloc().begin(), identities.get()->fieldloc().end());
          fieldloc.push_back(std::pair<int64_t, std::string>(identities.get()->width() - 1, std::to_string(j)));
          contents_[j].get()->setidentities(identities.get()->withfieldloc(fieldloc));
        }
      }
      else {
        Identities::FieldLoc original = identities.get()->fieldloc();
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identities::FieldLoc fieldloc(original.begin(), original.end());
          fieldloc.push_back(std::pair<int64_t, std::string>(identities.get()->width() - 1, recordlookup_.get()->at(j)));
          contents_[j].get()->setidentities(identities.get()->withfieldloc(fieldloc));
        }
      }
    }
    identities_ = identities;
  }

  const std::shared_ptr<Type> RecordArray::type() const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto item : contents_) {
      types.push_back(item.get()->type());
    }
    return std::make_shared<RecordType>(parameters_, types, recordlookup_);
  }

  const std::shared_ptr<Content> RecordArray::astype(const std::shared_ptr<Type>& type) const {
    if (RecordType* raw = dynamic_cast<RecordType*>(type.get())) {
      std::vector<std::shared_ptr<Content>> contents;
      if (raw->recordlookup().get() == nullptr) {
        for (int64_t i = 0;  i < raw->numfields();  i++) {
          if (i >= numfields()) {
            throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring() + std::string(" because tuple lengths don't match"));
          }
          contents.push_back(contents_[(size_t)i].get()->astype(raw->field(i)));
        }
      }
      else {
        for (auto key : raw->keys()) {
          if (!haskey(key)) {
            throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring() + std::string(" because the array doesn't have key ") + util::quote(key, true));
          }
          contents.push_back(contents_[(size_t)fieldindex(key)].get()->astype(raw->field(key)));
        }
      }
      if (contents.empty()) {
        return std::make_shared<RecordArray>(identities_, type.get()->parameters(), length(), istuple());
      }
      else {
        return std::make_shared<RecordArray>(identities_, type.get()->parameters(), contents, raw->recordlookup());
      }
    }
    else {
      throw std::invalid_argument(classname() + std::string(" cannot be converted to type ") + type.get()->tostring());
    }
  }

  const std::string RecordArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (contents_.empty()) {
      out << " length=\"" << length_ << "\"";
    }
    out << ">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n");
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
      out << contents_[j].get()->tostring_part(indent + std::string("        "), "", "\n");
      out << indent << "    </field>\n";
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void RecordArray::tojson_part(ToJson& builder) const {
    int64_t rows = length();
    size_t cols = contents_.size();
    std::shared_ptr<util::RecordLookup> keys = recordlookup_;
    if (istuple()) {
      keys = std::make_shared<util::RecordLookup>();
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    check_for_iteration();
    builder.beginlist();
    for (int64_t i = 0;  i < rows;  i++) {
      builder.beginrecord();
      for (size_t j = 0;  j < cols;  j++) {
        builder.field(keys.get()->at(j).c_str());
        contents_[j].get()->getitem_at_nowrap(i).get()->tojson_part(builder);
      }
      builder.endrecord();
    }
    builder.endlist();
  }

  void RecordArray::nbytes_part(std::map<size_t, int64_t>& largest) const {
    for (auto x : contents_) {
      x.get()->nbytes_part(largest);
    }
    if (identities_.get() != nullptr) {
      identities_.get()->nbytes_part(largest);
    }
  }

  int64_t RecordArray::length() const {
    if (contents_.empty()) {
      return length_;
    }
    else {
      int64_t out = -1;
      for (auto x : contents_) {
        int64_t len = x.get()->length();
        if (out < 0  ||  out > len) {
          out = len;
        }
      }
      return out;
    }
  }

  const std::shared_ptr<Content> RecordArray::shallow_copy() const {
    if (contents_.empty()) {
      return std::make_shared<RecordArray>(identities_, parameters_, length(), istuple());
    }
    else {
      return std::make_shared<RecordArray>(identities_, parameters_, contents_, recordlookup_);
    }
  }

  const std::shared_ptr<Content> RecordArray::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto x : contents_) {
      contents.push_back(x.get()->deep_copy(copyarrays, copyindexes, copyidentities));
    }
    std::shared_ptr<Identities> identities = identities_;
    if (copyidentities  &&  identities_.get() != nullptr) {
      identities = identities_.get()->deep_copy();
    }
    if (contents.empty()) {
      return std::make_shared<RecordArray>(identities, parameters_, length(), istuple());
    }
    else {
      return std::make_shared<RecordArray>(identities, parameters_, contents, recordlookup_);
    }
  }

  void RecordArray::check_for_iteration() const {
    if (identities_.get() != nullptr  &&  identities_.get()->length() < length()) {
      util::handle_error(failure("len(identities) < len(array)", kSliceNone, kSliceNone), identities_.get()->classname(), nullptr);
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_nothing() const {
    return getitem_range_nowrap(0, 0);
  }

  const std::shared_ptr<Content> RecordArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    int64_t len = length();
    if (regular_at < 0) {
      regular_at += len;
    }
    if (!(0 <= regular_at  &&  regular_at < len)) {
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const std::shared_ptr<Content> RecordArray::getitem_at_nowrap(int64_t at) const {
    return std::make_shared<Record>(*this, at);
  }

  const std::shared_ptr<Content> RecordArray::getitem_range(int64_t start, int64_t stop) const {
    if (contents_.empty()) {
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length());
      return std::make_shared<RecordArray>(identities_, parameters_, regular_stop - regular_start, istuple());
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_range(start, stop));
      }
      return std::make_shared<RecordArray>(identities_, parameters_, contents, recordlookup_);
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    if (contents_.empty()) {
      return std::make_shared<RecordArray>(identities_, parameters_, stop - start, istuple());
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_range_nowrap(start, stop));
      }
      return std::make_shared<RecordArray>(identities_, parameters_, contents, recordlookup_);
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_field(const std::string& key) const {
    return field(key).get()->getitem_range_nowrap(0, length());
  }

  const std::shared_ptr<Content> RecordArray::getitem_fields(const std::vector<std::string>& keys) const {
    RecordArray out(identities_, parameters_, length(), istuple());
    if (istuple()) {
      for (auto key : keys) {
        out.append(field(key).get()->getitem_range_nowrap(0, length()));
      }
    }
    else {
      for (auto key : keys) {
        out.append(field(key).get()->getitem_range_nowrap(0, length()), key);
      }
    }
    return out.shallow_copy();
  }

  const std::shared_ptr<Content> RecordArray::carry(const Index64& carry) const {
    if (contents_.empty()) {
      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }
      return std::make_shared<RecordArray>(identities, parameters_, carry.length(), istuple());
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->carry(carry));
      }
      std::shared_ptr<Identities> identities(nullptr);
      if (identities_.get() != nullptr) {
        identities = identities_.get()->getitem_carry_64(carry);
      }
      return std::make_shared<RecordArray>(identities, parameters_, contents, recordlookup_);
    }
  }

  bool RecordArray::purelist_isregular() const {
    return true;
  }

  int64_t RecordArray::purelist_depth() const {
    return 1;
  }

  const std::pair<int64_t, int64_t> RecordArray::minmax_depth() const {
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

  int64_t RecordArray::numfields() const {
    return (int64_t)contents_.size();
  }

  int64_t RecordArray::fieldindex(const std::string& key) const {
    return util::fieldindex(recordlookup_, key, numfields());
  }

  const std::string RecordArray::key(int64_t fieldindex) const {
    return util::key(recordlookup_, fieldindex, numfields());
  }

  bool RecordArray::haskey(const std::string& key) const {
    return util::haskey(recordlookup_, key, numfields());
  }

  const std::vector<std::string> RecordArray::keys() const {
    return util::keys(recordlookup_, numfields());
  }

  const Index64 RecordArray::count64() const {
    throw std::invalid_argument("RecordArray cannot be counted, because records are not lists");
  }

  const std::shared_ptr<Content> RecordArray::count(int64_t axis) const {
    if (axis != 0) {
      throw std::runtime_error("FIXME: RecordArray::count(axis != 0)");
    }
    std::vector<std::shared_ptr<Content>> contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->count(axis));
    }
    return std::make_shared<RecordArray>(identities_, parameters_, contents, recordlookup_);
  }

  const std::shared_ptr<Content> RecordArray::flatten(int64_t axis) const {
    std::vector<std::shared_ptr<Content>> contents;
    for (auto content : contents_) {
      contents.push_back(content.get()->flatten(axis));
    }
    return std::make_shared<RecordArray>(identities_, parameters_, contents, recordlookup_);
  }

  const std::shared_ptr<Content> RecordArray::field(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(std::string("fieldindex ") + std::to_string(fieldindex) + std::string(" for record with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    return contents_[(size_t)fieldindex];
  }

  const std::shared_ptr<Content> RecordArray::field(const std::string& key) const {
    return contents_[(size_t)fieldindex(key)];
  }

  const std::vector<std::shared_ptr<Content>> RecordArray::fields() const {
    return std::vector<std::shared_ptr<Content>>(contents_);
  }

  const std::vector<std::pair<std::string, std::shared_ptr<Content>>> RecordArray::fielditems() const {
    std::vector<std::pair<std::string, std::shared_ptr<Content>>> out;
    if (istuple()) {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(std::to_string(j), contents_[j]));
      }
    }
    else {
      size_t cols = contents_.size();
      for (size_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(recordlookup_.get()->at(j), contents_[j]));
      }
    }
    return out;
  }

  const RecordArray RecordArray::astuple() const {
    return RecordArray(identities_, parameters_, contents_);
  }

  void RecordArray::append(const std::shared_ptr<Content>& content, const std::string& key) {
    if (recordlookup_.get() == nullptr) {
      recordlookup_ = util::init_recordlookup(numfields());
    }
    contents_.push_back(content);
    recordlookup_.get()->push_back(key);
  }

  void RecordArray::append(const std::shared_ptr<Content>& content) {
    if (recordlookup_.get() == nullptr) {
      contents_.push_back(content);
    }
    else {
      append(content, std::to_string(numfields()));
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    Slice emptytail;
    emptytail.become_sealed();

    if (head.get() == nullptr) {
      return shallow_copy();
    }
    else if (SliceField* field = dynamic_cast<SliceField*>(head.get())) {
      std::shared_ptr<Content> out = getitem_next(*field, emptytail, advanced);
      return out.get()->getitem_next(nexthead, nexttail, advanced);
    }
    else if (SliceFields* fields = dynamic_cast<SliceFields*>(head.get())) {
      std::shared_ptr<Content> out = getitem_next(*fields, emptytail, advanced);
      return out.get()->getitem_next(nexthead, nexttail, advanced);
    }
    else if (contents_.empty()) {
      RecordArray out(Identities::none(), parameters_, length(), istuple());
      return out.getitem_next(nexthead, nexttail, advanced);
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_next(head, emptytail, advanced));
      }
      util::Parameters parameters;
      if (head.get()->preserves_type(advanced)) {
        parameters = parameters_;
      }
      RecordArray out(Identities::none(), parameters, contents, recordlookup_);
      return out.getitem_next(nexthead, nexttail, advanced);
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(std::string("undefined operation: RecordArray::getitem_next(at)"));
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(std::string("undefined operation: RecordArray::getitem_next(range)"));
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(std::string("undefined operation: RecordArray::getitem_next(array)"));
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_field(field.key()).get()->getitem_next(nexthead, nexttail, advanced);
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    std::shared_ptr<SliceItem> nexthead = tail.head();
    Slice nexttail = tail.tail();
    return getitem_fields(fields.keys()).get()->getitem_next(nexthead, nexttail, advanced);
  }

}
