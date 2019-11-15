// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"
#include "awkward/type/RegularType.h"

#include "awkward/array/RegularArray.h"

namespace awkward {
  int64_t RegularArray::ndim() const {
    return (int64_t)shape_.size();
  }

  bool RegularArray::isempty() const {
    for (auto x : shape_) {
      if (x == 0) {
        return true;
      }
    }
    return false;
  }

  const std::string RegularArray::classname() const {
    return "RegularArray";
  }

  void RegularArray::setid() {
    throw std::runtime_error("setid");
  }
  void RegularArray::setid(const std::shared_ptr<Identity> id) {
    throw std::runtime_error("setid(const std::shared_ptr<Identity> id)");
  }

  const std::string RegularArray::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname() << " shape=\"";
    for (int64_t i = 0;  i < ndim();  i++) {
      if (i != 0) {
        out << " ";
      }
      out << shape_[i];
    }
    out << "\">\n";
    if (id_.get() != nullptr) {
      out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
    }
    out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void RegularArray::tojson_part(ToJson& builder) const {
    throw std::runtime_error("tojson_part");
  }

  const std::shared_ptr<Type> RegularArray::type_part() const {
    assert(shape_.size() != 0);
    std::vector<int64_t> shape(shape_.begin() + 1, shape_.end());
    return std::shared_ptr<Type>(new RegularType(shape, content_.get()->type_part()));
  }

  int64_t RegularArray::length() const {
    assert(shape_.size() != 0);
    return shape_[0];
  }

  const std::shared_ptr<Content> RegularArray::shallow_copy() const {
    return std::shared_ptr<Content>(new RegularArray(id_, shape_, content_));
  }

  void RegularArray::checksafe() const { }

  const std::shared_ptr<Content> RegularArray::getitem_at(int64_t at) const {
    throw std::runtime_error("getitem_at");
  }

  const std::shared_ptr<Content> RegularArray::getitem_at_unsafe(int64_t at) const {
    throw std::runtime_error("getitem_at_unsafe");
  }

  const std::shared_ptr<Content> RegularArray::getitem_range(int64_t start, int64_t stop) const {
    throw std::runtime_error("getitem_range");
  }

  const std::shared_ptr<Content> RegularArray::getitem_range_unsafe(int64_t start, int64_t stop) const {
    throw std::runtime_error("RegularArray::getitem_range_unsafe");
  }

  const std::shared_ptr<Content> RegularArray::getitem(const Slice& where) const {
    throw std::runtime_error("RegularArray::getitem");
  }

  const std::shared_ptr<Content> RegularArray::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const {
    throw std::runtime_error("RegularArray::getitem_next");
  }

  const std::shared_ptr<Content> RegularArray::carry(const Index64& carry) const {
    throw std::runtime_error("RegularArray::carry");
  }

  const std::pair<int64_t, int64_t> RegularArray::minmax_depth() const {
    std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
    return std::pair<int64_t, int64_t>(content_depth.first + ndim(), content_depth.second + ndim());
  }
}
