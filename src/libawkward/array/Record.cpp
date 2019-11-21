// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
// #include "awkward/type/RecordType.h"

#include "awkward/array/Record.h"

namespace awkward {
  const std::string Record::classname() const {
    throw std::runtime_error("Record::classname");
  }

  const std::shared_ptr<Identity> Record::id() const {
    std::shared_ptr<Identity> recid = recordarray_.id();
    if (recid.get() == nullptr) {
      return recid;
    }
    else {
      return recid.get()->getitem_range_nowrap(where_, where_ + 1);
    }
  }

  void Record::setid() {
    throw std::runtime_error("Record::setid");
  }

  void Record::setid(const std::shared_ptr<Identity> id) {
    throw std::runtime_error("Record::setid");
  }

  const std::string Record::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    throw std::runtime_error("Record::tostring_part");
  }

  void Record::tojson_part(ToJson& builder) const {
    throw std::runtime_error("Record::tojson_part");
  }

  const std::shared_ptr<Type> Record::type_part() const {
    throw std::runtime_error("Record::type_part");
  }

  int64_t Record::length() const {
    throw std::runtime_error("Record::length");
  }

  const std::shared_ptr<Content> Record::shallow_copy() const {
    throw std::runtime_error("Record::shallow_copy");
  }

  void Record::check_for_iteration() const {
    throw std::runtime_error("Record::check_for_iteration");
  }

  const std::shared_ptr<Content> Record::getitem_nothing() const {
    throw std::runtime_error("Record::getitem_nothing");
  }

  const std::shared_ptr<Content> Record::getitem_at(int64_t at) const {
    throw std::runtime_error("Record::getitem_at");
  }

  const std::shared_ptr<Content> Record::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("Record::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> Record::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("Record::getitem_range");
  }

  const std::shared_ptr<Content> Record::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("Record::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> Record::getitem(const Slice& where) const {
    throw std::runtime_error("Record::getitem");
  }

  const std::shared_ptr<Content> Record::carry(const Index64& carry) const {
    throw std::runtime_error("Record::carry");
  }

  const std::pair<int64_t, int64_t> Record::minmax_depth() const {
    throw std::runtime_error("Record::minmax_depth");
  }

  int64_t Record::numfields() const {
    throw std::runtime_error("Record::numfields");
  }

  int64_t Record::index(const std::string& key) const {
    throw std::runtime_error("Record::index");
  }

  const std::string Record::key(int64_t index) const {
    throw std::runtime_error("Record::key");
  }

  bool Record::has(const std::string& key) const {
    throw std::runtime_error("Record::has");
  }

  const std::vector<std::string> Record::aliases(int64_t index) const {
    throw std::runtime_error("Record::aliases");
  }

  const std::vector<std::string> Record::aliases(const std::string& key) const {
    throw std::runtime_error("Record::aliases");
  }

  const std::shared_ptr<Content> Record::field(int64_t index) const {
    throw std::runtime_error("Record::field");
  }

  const std::shared_ptr<Content> Record::field(const std::string& key) const {
    throw std::runtime_error("Record::field");
  }


  const std::shared_ptr<Content> Record::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("Record::getitem_next");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("Record::getitem_next");
  }

  const std::shared_ptr<Content> Record::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("Record::getitem_next");
  }

}
