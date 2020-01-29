// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ArrayType.h"

#include "awkward/array/None.h"

namespace awkward {
  None::None()
      : Content(Identities::none(), util::Parameters()) { }

  bool None::isscalar() const {
    return true;
  }

  const std::string None::classname() const {
    return "None";
  }

  void None::setidentities(const std::shared_ptr<Identities>& identities) {
    throw std::runtime_error("undefined operation: None::setidentities(identities)");
  }

  void None::setidentities() {
    throw std::runtime_error("undefined operation: None::setidentities()");
  }

  const std::shared_ptr<Type> None::type() const {
    throw std::runtime_error("undefined operation: None::type()");
  }

  const std::shared_ptr<Content> None::astype(const std::shared_ptr<Type>& type) const {
    throw std::runtime_error("undefined operation: None::astype()");
  }

  const std::string None::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << "/>" << post;
    return out.str();
  }

  void None::tojson_part(ToJson& builder) const {
    builder.null();
  }

  void None::nbytes_part(std::map<size_t, int64_t>& largest) const {
    throw std::runtime_error("undefined operation: None::nbytes_part");
  }

  int64_t None::length() const {
    return -1;
  }

  const std::shared_ptr<Content> None::shallow_copy() const {
    return std::make_shared<None>();
  }

  const std::shared_ptr<Content> None::deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const {
    return std::make_shared<None>();
  }

  void None::check_for_iteration() const { }

  const std::shared_ptr<Content> None::getitem_nothing() const {
    throw std::runtime_error("undefined operation: None::getitem_nothing");
  }

  const std::shared_ptr<Content> None::getitem_at(int64_t at) const {
    throw std::runtime_error("undefined operation: None::getitem_at");
  }

  const std::shared_ptr<Content> None::getitem_at_nowrap(int64_t at) const {
    throw std::runtime_error("undefined operation: None::getitem_at_nowrap");
  }

  const std::shared_ptr<Content> None::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("undefined operation: None::getitem_range");
  }

  const std::shared_ptr<Content> None::getitem_range_nowrap(int64_t start, int64_t stop) const {
    throw std::runtime_error("undefined operation: None::getitem_range_nowrap");
  }

  const std::shared_ptr<Content> None::getitem_field(const std::string& key) const {
    throw std::runtime_error("undefined operation: None::getitem_field");
  }

  const std::shared_ptr<Content> None::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::runtime_error("undefined operation: None::getitem_fields");
  }

  const std::shared_ptr<Content> None::carry(const Index64& carry) const {
    throw std::runtime_error("undefined operation: None::carry");
  }

  bool None::purelist_isregular() const {
    throw std::runtime_error("undefined operation: None::purelist_isregular");
  }

  int64_t None::purelist_depth() const {
    throw std::runtime_error("undefined operation: None::purelist_depth");
  }

  const std::pair<int64_t, int64_t> None::minmax_depth() const {
    throw std::runtime_error("undefined operation: None::minmax_depth");
  }

  int64_t None::numfields() const {
    throw std::runtime_error("undefined operation: None::numfields");
  }

  int64_t None::fieldindex(const std::string& key) const {
    throw std::runtime_error("undefined operation: None::fieldindex");
  }

  const std::string None::key(int64_t fieldindex) const {
    throw std::runtime_error("undefined operation: None::key");
  }

  bool None::haskey(const std::string& key) const {
    throw std::runtime_error("undefined operation: None::haskey");
  }

  const std::vector<std::string> None::keys() const {
    throw std::runtime_error("undefined operation: None::keys");
  }

  const Index64 None::count64() const {
    throw std::runtime_error("undefined operation: None::count64");
  }

  const std::shared_ptr<Content> None::count(int64_t axis) const {
    throw std::runtime_error("undefined operation: None::count");
  }

  const std::shared_ptr<Content> None::flatten(int64_t axis) const {
    throw std::runtime_error("undefined operation: None::flatten");
  }

  bool None::mergeable(const std::shared_ptr<Content>& other, bool mergebool) const {
    throw std::runtime_error("undefined operation: None::mergeable");
  }

  const std::shared_ptr<Content> None::merge(const std::shared_ptr<Content>& other) const {
    throw std::runtime_error("undefined operation: None::merge");
  }

  const std::shared_ptr<Content> None::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(SliceAt)");
  }

  const std::shared_ptr<Content> None::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(SliceRange)");
  }

  const std::shared_ptr<Content> None::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(SliceArray64)");
  }

  const std::shared_ptr<Content> None::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(SliceField)");
  }

  const std::shared_ptr<Content> None::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("undefined operation: None::getitem_next(SliceFields)");
  }

  const std::shared_ptr<Content> none = std::make_shared<None>();
}
