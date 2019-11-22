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
    throw std::runtime_error("RecordArray::setid");
  }

  void RecordArray::setid(const std::shared_ptr<Identity> id) {
    throw std::runtime_error("RecordArray::setid");
  }

  const std::string RecordArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << ">\n";
    if (id_.get() != nullptr) {
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    for (size_t j = 0;  j < contents_.size();  j++) {
      out << indent << "    <field index=\"" << j << "\"";
      if (reverselookup_.get() != nullptr) {
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
    if (keys.get() == nullptr) {
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

  const std::shared_ptr<Type> RecordArray::type_part() const {
    std::vector<std::shared_ptr<Type>> types;
    for (auto item : contents_) {
      types.push_back(item.get()->type_part());
    }
    return std::shared_ptr<Type>(new RecordType(types, lookup_, reverselookup_));
  }

  int64_t RecordArray::length() const {
    int64_t out = -1;
    for (auto x : contents_) {
      int64_t len = x.get()->length();
      if (out < 0  ||  out > len) {
        out = len;
      }
    }
    return out;
  }

  const std::shared_ptr<Content> RecordArray::shallow_copy() const {
    return std::shared_ptr<Content>(new RecordArray(id_, contents_, lookup_, reverselookup_));
  }

  void RecordArray::check_for_iteration() const {
    throw std::runtime_error("RecordArray::check_for_iteration");
  }

  const std::shared_ptr<Content> RecordArray::getitem_nothing() const {
    throw std::runtime_error("RecordArray::getitem_nothing");
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
    throw std::runtime_error("RecordArray::getitem_range");
  }

  const std::shared_ptr<Content> RecordArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("RecordArray::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> RecordArray::carry(const Index64& carry) const {
    throw std::runtime_error("RecordArray::carry");
  }

  const std::pair<int64_t, int64_t> RecordArray::minmax_depth() const {
    throw std::runtime_error("RecordArray::minmax_depth");
  }

  int64_t RecordArray::numfields() const {
    return (int64_t)contents_.size();
  }

  int64_t RecordArray::index(const std::string& key) const {
    int64_t out = -1;
    if (lookup_.get() != nullptr) {
      try {
        out = (int64_t)lookup_.get()->at(key);
      }
      catch (std::out_of_range err) { }
      if (out != -1  &&  out >= numfields()) {
        throw std::invalid_argument(std::string("key \"") + key + std::string("\" points to tuple index ") + std::to_string(out) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
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
        throw std::invalid_argument(std::string("key interpreted as index ") + key + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
      }
    }
    return out;
  }

  const std::string RecordArray::key(int64_t index) const {
    if (index >= numfields()) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    if (reverselookup_.get() != nullptr) {
      return reverselookup_.get()->at((size_t)index);
    }
    else {
      return std::to_string(index);
    }
  }

  bool RecordArray::has(const std::string& key) const {
    try {
      index(key);
    }
    catch (std::invalid_argument err) {
      return false;
    }
    return true;
  }

  const std::vector<std::string> RecordArray::aliases(int64_t index) const {
    std::vector<std::string> out;
    std::string _default = std::to_string(index);
    bool has_default = false;
    if (lookup_.get() != nullptr) {
      for (auto pair : *lookup_.get()) {
        if (pair.second == index) {
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

  const std::vector<std::string> RecordArray::aliases(const std::string& key) const {
    return aliases(index(key));
  }

  const std::shared_ptr<Content> RecordArray::field(int64_t index) const {
    if (index >= numfields()) {
      throw std::invalid_argument(std::string("index ") + std::to_string(index) + std::string(" for RecordArray with only " + std::to_string(numfields()) + std::string(" fields")));
    }
    return contents_[(size_t)index];
  }

  const std::shared_ptr<Content> RecordArray::field(const std::string& key) const {
    return contents_[(size_t)index(key)];
  }

  const std::vector<std::string> RecordArray::keys() const {
    std::vector<std::string> out;
    if (reverselookup_.get() == nullptr) {
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

  const std::vector<std::shared_ptr<Content>> RecordArray::values() const {
    return std::vector<std::shared_ptr<Content>>(contents_);
  }

  const std::vector<std::pair<std::string, std::shared_ptr<Content>>> RecordArray::items() const {
    std::vector<std::pair<std::string, std::shared_ptr<Content>>> out;
    if (reverselookup_.get() == nullptr) {
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

  void RecordArray::append(const std::shared_ptr<Content>& content, const std::string& key) {
    size_t j = contents_.size();
    append(content);
    setkey(j, key);
  }

  void RecordArray::append(const std::shared_ptr<Content>& content) {
    if (reverselookup_.get() != nullptr) {
      reverselookup_.get()->push_back(std::to_string(contents_.size()));
    }
    contents_.push_back(content);
  }

  void RecordArray::setkey(int64_t index, const std::string& fieldname) {
    if (lookup_.get() == nullptr) {
      lookup_ = std::shared_ptr<Lookup>(new Lookup);
      reverselookup_ = std::shared_ptr<ReverseLookup>(new ReverseLookup);
      for (size_t j = 0;  j < contents_.size();  j++) {
        reverselookup_.get()->push_back(std::to_string(j));
      }
    }
    (*lookup_.get())[fieldname] = (size_t)index;
    (*reverselookup_.get())[(size_t)index] = fieldname;
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("RecordArray::getitem_next(at)");
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("RecordArray::getitem_next(range)");
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("RecordArray::getitem_next(array)");
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: RecordArray::getitem_next(field)");
  }

  const std::shared_ptr<Content> RecordArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: RecordArray::getitem_next(fields)");
  }

}
