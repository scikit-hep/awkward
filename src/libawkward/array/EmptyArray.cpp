// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ArrayType.h"

#include "awkward/array/EmptyArray.h"

namespace awkward {
  EmptyArray::EmptyArray(const std::shared_ptr<Identity>& id, const std::shared_ptr<Type>& type)
      : Content(id, type) {
    if (type_.get() != nullptr) {
      checktype();
    }
  }

  const std::string EmptyArray::classname() const {
    return "EmptyArray";
  }

  void EmptyArray::setid(const std::shared_ptr<Identity>& id) {
    if (id.get() != nullptr  &&  length() != id.get()->length()) {
      util::handle_error(failure("content and its id must have the same length", kSliceNone, kSliceNone), classname(), id_.get());
    }
    id_ = id;
  }

  void EmptyArray::setid() { }

  const std::shared_ptr<Type> EmptyArray::type() const {
    if (type_.get() != nullptr) {
      return type_;
    }
    else {
      return std::make_shared<UnknownType>(util::Parameters());
    }
  }

  const std::shared_ptr<Content> EmptyArray::astype(const std::shared_ptr<Type>& type) const {
    return type.get()->empty();
  }

  const std::string EmptyArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (id_.get() == nullptr  &&  parameters_.empty()  &&  type_.get() == nullptr) {
      out << "/>" << post;
    }
    else {
      out << ">\n";
      if (id_.get() != nullptr) {
        out << id_.get()->tostring_part(indent + std::string("    "), "", "\n") << indent << "</" << classname() << ">" << post;
      }
      if (!parameters_.empty()) {
        out << parameters_tostring(indent + std::string("    "), "", "\n");
      }
      if (type_.get() != nullptr) {
        out << indent << "    <type>" + type().get()->tostring() + "</type>\n";
      }
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  void EmptyArray::tojson_part(ToJson& builder) const {
    builder.beginlist();
    builder.endlist();
  }

  int64_t EmptyArray::length() const {
    return 0;
  }

  const std::shared_ptr<Content> EmptyArray::shallow_copy() const {
    return std::make_shared<EmptyArray>(id_, type_);
  }

  void EmptyArray::check_for_iteration() const { }

  const std::shared_ptr<Content> EmptyArray::getitem_nothing() const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_at(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_at_nowrap(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_range(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_field(const std::string& key) const {
    throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
  }

  const std::shared_ptr<Content> EmptyArray::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(std::string("cannot slice ") + classname() + std::string(" by field name"));
  }

  const std::shared_ptr<Content> EmptyArray::carry(const Index64& carry) const {
    return shallow_copy();
  }

  const std::pair<int64_t, int64_t> EmptyArray::minmax_depth() const {
    return std::pair<int64_t, int64_t>(1, 1);
  }

  int64_t EmptyArray::numfields() const { return -1; }

  int64_t EmptyArray::fieldindex(const std::string& key) const {
    throw std::invalid_argument("array contains no Records");
  }

  const std::string EmptyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument("array contains no Records");
  }

  bool EmptyArray::haskey(const std::string& key) const {
    throw std::invalid_argument("array contains no Records");
  }

  const std::vector<std::string> EmptyArray::keys() const {
    throw std::invalid_argument("array contains no Records");
  }

  void EmptyArray::checktype() const { }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(field.tostring() + std::string(" is not a valid slice type for ") + classname());
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(fields.tostring() + std::string(" is not a valid slice type for ") + classname());
  }

}
