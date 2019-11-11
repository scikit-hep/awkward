// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

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

  int64_t EmptyArray::length() const {
    return 0;
  }

  const std::shared_ptr<Content> EmptyArray::shallow_copy() const {
    return std::shared_ptr<Content>(new EmptyArray(id_));
  }

  void EmptyArray::checksafe() const { }

  const std::shared_ptr<Content> EmptyArray::getitem_at(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_at_unsafe(int64_t at) const {
    util::handle_error(failure("index out of range", kSliceNone, at), classname(), id_.get());
    return std::shared_ptr<Content>(nullptr);  // make Windows compiler happy
  }

  const std::shared_ptr<Content> EmptyArray::getitem_range(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_range_unsafe(int64_t start, int64_t stop) const {
    return shallow_copy();
  }

  const std::shared_ptr<Content> EmptyArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    if (head.get() == nullptr) {
      throw std::runtime_error("FIXME null");
    }

    else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
      throw std::runtime_error("FIXME at");
    }

    else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
      throw std::runtime_error("FIXME range");
    }

    else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
      return getitem_ellipsis(tail, advanced);
    }

    else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
      return getitem_newaxis(tail, advanced);
    }

    else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
      if (advanced.length() == 0) {
        throw std::runtime_error("FIXME array");
      }
      else {
        throw std::runtime_error("FIXME array advanced");
      }
    }

    else {
      throw std::runtime_error("unrecognized slice item type");
    }
  }

  const std::shared_ptr<Content> EmptyArray::carry(const Index64& carry) const {
    throw std::runtime_error("FIXME carry");
  }

  const std::pair<int64_t, int64_t> EmptyArray::minmax_depth() const { return std::pair<int64_t, int64_t>(1, 1); }
}
