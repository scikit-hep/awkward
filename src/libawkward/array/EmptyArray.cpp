// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/type/UnknownType.h"
#include "awkward/type/ArrayType.h"
#include "awkward/array/NumpyArray.h"

#include "awkward/array/EmptyArray.h"

namespace awkward {
  EmptyArray::EmptyArray(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters)
      : Content(identities, parameters) { }

  const std::string EmptyArray::classname() const {
    return "EmptyArray";
  }

  void EmptyArray::setidentities(const std::shared_ptr<Identities>& identities) {
    if (identities.get() != nullptr  &&  length() != identities.get()->length()) {
      util::handle_error(failure("content and its identities must have the same length", kSliceNone, kSliceNone), classname(), identities_.get());
    }
    identities_ = identities;
  }

  void EmptyArray::setidentities() { }

  const std::shared_ptr<Type> EmptyArray::type() const {
    return std::make_shared<UnknownType>(parameters_);
  }

  const std::shared_ptr<Content> EmptyArray::astype(const std::shared_ptr<Type>& type) const {
    return type.get()->empty();
  }

  const std::string EmptyArray::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname();
    if (identities_.get() == nullptr  &&  parameters_.empty()) {
      out << "/>" << post;
    }
    else {
      out << ">\n";
      if (identities_.get() != nullptr) {
        out << identities_.get()->tostring_part(indent + std::string("    "), "", "\n") << indent << "</" << classname() << ">" << post;
      }
      if (!parameters_.empty()) {
        out << parameters_tostring(indent + std::string("    "), "", "\n");
      }
      out << indent << "</" << classname() << ">" << post;
    }
    return out.str();
  }

  void EmptyArray::tojson_part(ToJson& builder) const {
    check_for_iteration();
    builder.beginlist();
    builder.endlist();
  }

  int64_t EmptyArray::length() const {
    return 0;
  }

  const std::shared_ptr<Content> EmptyArray::shallow_copy() const {
    return std::make_shared<EmptyArray>(identities_, parameters_);
  }

  void EmptyArray::check_for_iteration() const { }

  const std::shared_ptr<Content> EmptyArray::getitem_nothing() const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_at(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_at_nowrap(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), identities_.get());
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
    throw std::invalid_argument(std::string("key ") + util::quote(key, true) + std::string(" does not exist (data might not be records)"));
  }

  const std::string EmptyArray::key(int64_t fieldindex) const {
    throw std::invalid_argument(std::string("fieldindex \"") + std::to_string(fieldindex) + std::string("\" does not exist (data might not be records)"));
  }

  bool EmptyArray::haskey(const std::string& key) const {
    return false;
  }

  const std::vector<std::string> EmptyArray::keys() const {
    return std::vector<std::string>();
  }

  int64_t EmptyArray::list_depth() const {
    throw std::runtime_error("FIXME: EmptyArray::list_depth() is not implemented");
  }

  const Index64 EmptyArray::count64() const {
    return Index64(0);
  }

  const std::shared_ptr<Content> EmptyArray::count(int64_t axis) const {
    Index64 tocount = count64();
    std::vector<ssize_t> shape({ (ssize_t)tocount.length() });
    std::vector<ssize_t> strides({ (ssize_t)sizeof(int64_t) });
#ifdef _MSC_VER
    std::string format = "q";
#else
    std::string format = "l";
#endif
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), tocount.ptr(), shape, strides, 0, sizeof(int64_t), format);
  }

  const std::shared_ptr<Content> EmptyArray::flatten(int64_t axis) const {
    return std::make_shared<EmptyArray>(Identities::none(), util::Parameters());
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const {
    util::handle_error(failure("too many dimensions in slice", kSliceNone, kSliceNone), classname(), identities_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceField& field, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(field.tostring() + std::string(" is not a valid slice type for ") + classname());
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const SliceFields& fields, const Slice& tail, const Index64& advanced) const {
    throw std::invalid_argument(fields.tostring() + std::string(" is not a valid slice type for ") + classname());
  }

}
