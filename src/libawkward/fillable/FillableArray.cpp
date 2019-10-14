// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/type/ArrayType.h"

#include "awkward/fillable/FillableArray.h"

namespace awkward {
  std::string FillableArray::tostring() const {
    std::stringstream out;
    out << "<FillableArray length=\"" << length() << "\" type=\"" << type().get()->tostring() << "\"/>";
    return out.str();
  }

  int64_t FillableArray::length() const {
    return fillable_.get()->length();
  }

  void FillableArray::clear() {
    fillable_.get()->clear();
  }

  const std::shared_ptr<Type> FillableArray::type() const {
    return std::shared_ptr<Type>(new ArrayType(fillable_.get()->length(), fillable_.get()->type()));
  }

  const std::shared_ptr<Content> FillableArray::layout() const {
    return fillable_.get()->layout();
  }

  void FillableArray::boolean(bool x) {
    fillable_.get()->boolean(x);
  }
}
