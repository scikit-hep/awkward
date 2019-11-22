// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/RecordType.h"

#include "awkward/array/Record.h"

namespace awkward {
  bool Record::isscalar() const {
    return true;
  }

  const std::string Record::classname() const {
    return "Record";
  }

  const std::shared_ptr<Identity> Record::id() const {
    std::shared_ptr<Identity> recid = recordarray_.id();
    if (recid.get() == nullptr) {
      return recid;
    }
    else {
      return recid.get()->getitem_range_nowrap(at_, at_ + 1);
    }
  }

  void Record::setid() {
    throw std::runtime_error("undefined operation: Record::setid");
  }

  void Record::setid(const std::shared_ptr<Identity> id) {
    throw std::runtime_error("undefined operation: Record::setid");
  }

  const std::string Record::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " at=\"" << at_ << "\">\n";
    out << recordarray_.tostring_part(indent + std::string("    "), "", "\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void Record::tojson_part(ToJson& builder) const {
    size_t cols = contents_.size();
    std::shared_ptr<RecordArray::ReverseLookup> keys = recordarray_.reverselookup();
    if (keys.get() == nullptr) {
      keys = std::shared_ptr<RecordArray::ReverseLookup>(new RecordArray::ReverseLookup);
      for (size_t j = 0;  j < cols;  j++) {
        keys.get()->push_back(std::to_string(j));
      }
    }
    std::vector<std::shared_ptr<Content>> contents = recordarray_.contents();
    builder.beginrec();
    for (size_t j = 0;  j < cols;  j++) {
      builder.fieldkey(keys.get()->at(j).c_str());
      contents[j].get()->getitem_at_nowrap(at_).get()->tojson_part(builder);
    }
    builder.endrec();
  }

  const std::shared_ptr<Type> Record::type_part() const {
    return recordarray_.type_part();
  }

  int64_t Record::length() const {
    return -1;   // just like NumpyArray with ndim == 0, which is also a scalar
  }

  const std::shared_ptr<Content> Record::shallow_copy() const {
    return std::shared_ptr<Content>(new Record(recordarray_, at_));
  }

  void Record::check_for_iteration() const {
    throw std::runtime_error("FIXME: Record::check_for_iteration");
  }

  const std::shared_ptr<Content> Record::getitem_nothing() const {
    throw std::runtime_error("FIXME: Record::getitem_nothing");
  }

  const std::shared_ptr<Content> Record::getitem_at(int64_t at) const {
    throw std::runtime_error("FIXME: Record::getitem_at");
  }

  const std::shared_ptr<Content> Record::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("FIXME: Record::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> Record::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: Record::getitem_range");
  }

  const std::shared_ptr<Content> Record::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("FIXME: Record::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> Record::getitem(const Slice& where) const {
    throw std::runtime_error("FIXME: Record::getitem");
  }

  const std::shared_ptr<Content> Record::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME: Record::carry");
  }

  const std::pair<int64_t, int64_t> Record::minmax_depth() const {
    throw std::runtime_error("FIXME: Record::minmax_depth");
  }

  int64_t Record::numfields() const {
    return recordarray_.numfields();
  }

  int64_t Record::index(const std::string& key) const {
    return recordarray_.index(key);
  }

  const std::string Record::key(int64_t index) const {
    return recordarray_.key(index);
  }

  bool Record::has(const std::string& key) const {
    return recordarray_.has(key);
  }

  const std::vector<std::string> Record::aliases(int64_t index) const {
    return recordarray_.aliases(index);
  }

  const std::vector<std::string> Record::aliases(const std::string& key) const {
    return recordarray_.aliases(key);
  }

  const std::shared_ptr<Content> Record::field(int64_t index) const {
    return recordarray_.field(index).get()->getitem_at_nowrap(at_);
  }

  const std::shared_ptr<Content> Record::field(const std::string& key) const {
    return recordarray_.field(key).get()->getitem_at_nowrap(at_);
  }

  const std::vector<std::string> Record::keys() const {
    return recordarray_.keys();
  }

  const std::vector<std::shared_ptr<Content>> Record::values() const {
    std::vector<std::shared_ptr<Content>> out;
    int64_t cols = numfields();
    for (int64_t j = 0;  j < cols;  j++) {
      out.push_back(recordarray_.field(j).get()->getitem_at_nowrap(at_));
    }
    return out;
  }

  const std::vector<std::pair<std::string, std::shared_ptr<Content>>> Record::items() const {
    std::vector<std::pair<std::string, std::shared_ptr<Content>>> out;
    std::shared_ptr<RecordArray::ReverseLookup> keys = recordarray_.reverselookup();
    if (keys.get() == nullptr) {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(std::to_string(j), recordarray_.field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    else {
      int64_t cols = numfields();
      for (int64_t j = 0;  j < cols;  j++) {
        out.push_back(std::pair<std::string, std::shared_ptr<Content>>(keys.get()->at((size_t)j), recordarray_.field(j).get()->getitem_at_nowrap(at_)));
      }
    }
    return out;
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: Record::getitem_next");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: Record::getitem_next");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("FIXME: Record::getitem_next");
  }

}
