// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/type/UnknownType.h"

#include "awkward/array/EmptyArray.h"

namespace awkward {
  const std::string EmptyArray::classname() const {
    return "EmptyArray";
  }

  void EmptyArray::setid(const std::shared_ptr<Identity> id) {
    if (id.get() != nullptr  &&  length() != id.get()->length()) {
      util::handle_error(failure("content and its id must have the same length", kSliceNone, kSliceNone), classname(), id_.get());
    }
    id_ = id;
  }

  void EmptyArray::setid() { }

  const std::string EmptyArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (id_.get() != nullptr) {
      out << ">\n" << id_.get()->tostring_part(indent + std::string("    "), "", "\n") << indent << "</" << classname() << ">" << post;
    }
    else {
      out << "/>" << post;
    }
    return out.str();
  }

  void EmptyArray::tojson_part(ToJson& builder) const {
    builder.beginlist();
    builder.endlist();
  }

  const std::shared_ptr<Type> EmptyArray::type_part() const {
    return std::shared_ptr<Type>(new UnknownType());
  }

  int64_t EmptyArray::length() const {
    return 0;
  }

  const std::shared_ptr<Content> EmptyArray::shallow_copy() const {
    return std::shared_ptr<Content>(new EmptyArray(id_));
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
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

}
