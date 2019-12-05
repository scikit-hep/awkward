// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/RecordType.h"
#include "awkward/array/Record.h"

#include "awkward/array/RecordArray.h"

namespace awkward {
  const std::string RecordArray::classname() const {
    return "RecordArray";
  }

  void RecordArray::setid() {
    int64_t len = length();
    if (len <= kMaxInt32) {
      Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, len);
      std::shared_ptr<Identity> newid(rawid);
      struct Error err = awkward_new_identity32(rawid->ptr().get(), len);
      util::handle_error(err, classname(), id_.get());
      setid(newid);
    }
    else {
      Identity64* rawid = new Identity64(Identity::newref(), Identity::FieldLoc(), 1, len);
      std::shared_ptr<Identity> newid(rawid);
      struct Error err = awkward_new_identity64(rawid->ptr().get(), len);
      util::handle_error(err, classname(), id_.get());
      setid(newid);
    }
  }

  void RecordArray::setid(const std::shared_ptr<Identity> id) {
    if (id.get() == nullptr) {
      for (auto content : contents_) {
        content.get()->setid(id);
      }
    }
    else {
      if (length() != id.get()->length()) {
        util::handle_error(failure("content and its id must have the same length", kSliceNone, kSliceNone), classname(), id_.get());
      }
      if (istuple()) {
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identity::FieldLoc fieldloc(id.get()->fieldloc().begin(), id.get()->fieldloc().end());
          fieldloc.push_back(std::pair<int64_t, std::string>(id.get()->width() - 1, std::to_string(j)));
          contents_[j].get()->setid(id.get()->withfieldloc(fieldloc));
        }
      }
      else {
        Identity::FieldLoc original = id.get()->fieldloc();
        for (size_t j = 0;  j < contents_.size();  j++) {
          Identity::FieldLoc fieldloc(original.begin(), original.end());
          fieldloc.push_back(std::pair<int64_t, std::string>(id.get()->width() - 1, reverselookup_.get()->at(j)));
          contents_[j].get()->setid(id.get()->withfieldloc(fieldloc));
        }
      }
    }
    id_ = id;
  }

  const std::string RecordArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (contents_.size() == 0) {
      out << " length=\"" << length_ << "\"";
    }
    out << ">\n";
    if (id_.get() != nullptr) {
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    for (size_t j = 0;  j < contents_.size();  j++) {
      out << indent << "    <field index=\"" << j << "\"";
      if (!istuple()) {
        out << " key=\"" << reverselookup_.get()->at(j) << "\">";
        for (auto pair : *lookup_.get()) {
          if (pair.second == j  &&  pair.first != reverselookup_.get()->at(j)) {
            out << "<alias>" << pair.first << "</alias>";
          }
        }
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
    std::shared_ptr<ReverseLookup> keys = reverselookup_;
    if (istuple()) {
      keys = std::shared_ptr<ReverseLookup>(new ReverseLookup);
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    builder.beginlist();
    for (int64_t i = 0;  i < rows;  i++) {
      builder.beginrec();
      for (size_t j = 0;  j < cols;  j++) {
        builder.fieldkey(keys.get()->at(j).c_str());
        contents_[j].get()->getitem_at_nowrap(i).get()->tojson_part(builder);
      }
      builder.endrec();
    }
    builder.endlist();
  }

  const std::shared_ptr<Type> RecordArray::innertype(bool bare) const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto item : contents_) {
      types.push_back(item.get()->innertype(bare));
    }
    return std::shared_ptr<Type>(new RecordType(types, lookup_, reverselookup_));
  }

  void RecordArray::settype(const std::shared_ptr<Type> type) {
    if (accepts(type)) {
      // FIXME: apply to descendants
      type_ = type;
    }
    else {
      throw std::invalid_argument(std::string("provided type is incompatible with array: ") + type.get()->compare(baretype()));
    }
  }

  bool RecordArray::accepts(const std::shared_ptr<Type> type) {
    std::shared_ptr<Type> check = type.get()->level();
    if (RecordType* raw = dynamic_cast<RecordType*>(check.get())) {
      // if (numfields() != t->numfields()) {
      //   return false;
      // }
      // if (reverselookup_.get() == nullptr) {
      //   if (t->reverselookup().get() != nullptr) {
      //     return false;
      //   }
      //   return true;
      // }
      // else {
      //   if (t->reverselookup().get() == nullptr) {
      //     return false;
      //   }
      //   if (lookup_.get()->size() != t->lookup().get()->size()) {
      //     return false;
      //   }
      //   for (auto pair : *lookup_.get()) {
      //     int64_t otherindex;
      //     try {
      //       otherindex = (int64_t)t->lookup().get()->at(pair.first);
      //     }
      //     catch (std::out_of_range err) {
      //       return false;
      //     }
      //   }
      //   return true;
      // }
      return true;
    }
    else {
      return false;
    }
  }

  int64_t RecordArray::length() const {
    if (contents_.size() == 0) {
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
    if (contents_.size() == 0) {
      return std::shared_ptr<Content>(new RecordArray(id_, type_, length(), istuple()));
    }
    else {
      return std::shared_ptr<Content>(new RecordArray(id_, type_, contents_, lookup_, reverselookup_));
    }
  }

  void RecordArray::check_for_iteration() const {
    if (id_.get() != nullptr  &&  id_.get()->length() < length()) {
      util::handle_error(failure("len(id) < len(array)", kSliceNone, kSliceNone), id_.get()->classname(), nullptr);
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
      util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const std::shared_ptr<Content> RecordArray::getitem_at_nowrap(int64_t at) const {
    return std::shared_ptr<Content>(new Record(*this, at));
  }

  const std::shared_ptr<Content> RecordArray::getitem_range(int64_t start, int64_t stop) const {
    if (contents_.size() == 0) {
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, true, start != Slice::none(), stop != Slice::none(), length());
      return std::shared_ptr<Content>(new RecordArray(id_, type_, regular_stop - regular_start, istuple()));
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_range(start, stop));
      }
      return std::shared_ptr<Content>(new RecordArray(id_, type_, contents, lookup_, reverselookup_));
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    if (contents_.size() == 0) {
      return std::shared_ptr<Content>(new RecordArray(id_, type_, stop - start, istuple()));
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_range_nowrap(start, stop));
      }
      return std::shared_ptr<Content>(new RecordArray(id_, type_, contents, lookup_, reverselookup_));
    }
  }

  const std::shared_ptr<Content> RecordArray::getitem_field(const std::string& key) const {
    return field(key).get()->getitem_range_nowrap(0, length());
  }

  const std::shared_ptr<Content> RecordArray::getitem_fields(const std::vector<std::string>& keys) const {
    std::shared_ptr<Type> type = Type::none();
    if (type_.get() != nullptr  &&  type_.get()->numfields() != -1  &&  util::subset(keys, type_.get()->keys())) {
      type = type_;
    }
    RecordArray out(id_, type, length(), istuple());
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
    if (contents_.size() == 0) {
      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_carry_64(carry);
      }
      return std::shared_ptr<Content>(new RecordArray(id, type_, carry.length(), istuple()));
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->carry(carry));
      }
      std::shared_ptr<Identity> id(nullptr);
      if (id_.get() != nullptr) {
        id = id_.get()->getitem_carry_64(carry);
      }
      return std::shared_ptr<Content>(new RecordArray(id, type_, contents, lookup_, reverselookup_));
    }
  }

  const std::pair<int64_t, int64_t> RecordArray::minmax_depth() const {
    if (contents_.size() == 0) {
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
    int64_t out = -1;
    if (!istuple()) {
      try {
        out = (int64_t)lookup_.get()->at(key);
      }
      catch (std::out_of_range err) { }
      if (out != -1  &&  out >= numfields()) {
        throw std::invalid_argument(std::string("key \"") + key + std::string("\" points to fieldindex ") + std::to_string(out) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
      }
    }
    if (out == -1) {
      try {
        out = (int64_t)std::stoi(key);
      }
      catch (std::invalid_argument err) {
        throw std::invalid_argument(std::string("key \"") + key + std::string("\" is not in RecordArray"));
      }
      if (out >= numfields()) {
        throw std::invalid_argument(std::string("key interpreted as fieldindex ") + key + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
      }
    }
    return out;
  }

  const std::string RecordArray::key(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(std::string("fieldindex ") + std::to_string(fieldindex) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    if (!istuple()) {
      return reverselookup_.get()->at((size_t)fieldindex);
    }
    else {
      return std::to_string(fieldindex);
    }
  }

  bool RecordArray::haskey(const std::string& key) const {
    try {
      fieldindex(key);
    }
    catch (std::invalid_argument err) {
      return false;
    }
    return true;
  }

  const std::vector<std::string> RecordArray::keyaliases(int64_t fieldindex) const {
    std::vector<std::string> out;
    std::string _default = std::to_string(fieldindex);
    bool has_default = false;
    if (!istuple()) {
      for (auto pair : *lookup_.get()) {
        if (pair.second == fieldindex) {
          out.push_back(pair.first);
          if (pair.first == _default) {
            has_default = true;
          }
        }
      }
    }
    if (!has_default) {
      out.push_back(_default);
    }
    return out;
  }

  const std::vector<std::string> RecordArray::keyaliases(const std::string& key) const {
    return keyaliases(fieldindex(key));
  }

  const std::vector<std::string> RecordArray::keys() const {
    std::vector<std::string> out;
    if (istuple()) {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::to_string(j));
      }
    }
    else {
      out.insert(out.end(), reverselookup_.get()->begin(), reverselookup_.get()->end());
    }
    return out;
  }

  const std::shared_ptr<Content> RecordArray::field(int64_t fieldindex) const {
    if (fieldindex >= numfields()) {
      throw std::invalid_argument(std::string("fieldindex ") + std::to_string(fieldindex) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
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
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(reverselookup_.get()->at(j), contents_[j]));
      }
    }
    return out;
  }

  const RecordArray RecordArray::astuple() const {
    RecordArray out(id_, Type::none(), contents_);
    if (type_.get() != nullptr  &&  type_.get()->numfields() != -1  &&  util::subset(out.keys(), type_.get()->keys())) {
      out.type_ = type_;
    }
    return out;
  }

  void RecordArray::append(const std::shared_ptr<Content>& content, const std::string& key) {
    size_t j = contents_.size();
    append(content);
    setkey(j, key);
  }

  void RecordArray::append(const std::shared_ptr<Content>& content) {
    if (!istuple()) {
      reverselookup_.get()->push_back(std::to_string(contents_.size()));
    }
    contents_.push_back(content);
  }

  void RecordArray::setkey(int64_t fieldindex, const std::string& fieldname) {
    if (istuple()) {
      lookup_ = std::shared_ptr<Lookup>(new Lookup);
      reverselookup_ = std::shared_ptr<ReverseLookup>(new ReverseLookup);
      for (size_t j = 0;  j < contents_.size();  j++) {
        reverselookup_.get()->push_back(std::to_string(j));
      }
    }
    (*lookup_.get())[fieldname] = (size_t)fieldindex;
    (*reverselookup_.get())[(size_t)fieldindex] = fieldname;
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
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
    else if (contents_.size() == 0) {
      RecordArray out(Identity::none(), type_, length(), istuple());
      return out.getitem_next(nexthead, nexttail, advanced);
    }
    else {
      std::vector<std::shared_ptr<Content>> contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->getitem_next(head, emptytail, advanced));
      }
      std::shared_ptr<Type> type = Type::none();
      if (head.get()->preserves_type(type_, advanced)) {
        type = type_;
      }
      RecordArray out(Identity::none(), type, contents, lookup_, reverselookup_);
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
