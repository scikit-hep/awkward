// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/type/ArrayType.h"

#include "awkward/fillable/FillableArray.h"

namespace awkward {
  const std::string FillableArray::tostring() const {
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

  const std::shared_ptr<Content> FillableArray::snapshot() const {
    return fillable_.get()->snapshot();
  }

  const std::shared_ptr<Content> FillableArray::getitem_at(int64_t at) const {
    return snapshot().get()->getitem_at(at);
  }

  const std::shared_ptr<Content> FillableArray::getitem_range(int64_t start, int64_t stop) const {
    return snapshot().get()->getitem_range(start, stop);
  }

  const std::shared_ptr<Content> FillableArray::getitem(const Slice& where) const {
    return snapshot().get()->getitem(where);
  }

  void FillableArray::null() {
    maybeupdate(fillable_.get()->null());
  }

  void FillableArray::boolean(bool x) {
    maybeupdate(fillable_.get()->boolean(x));
  }

  void FillableArray::maybeupdate(Fillable* tmp) {
    if (tmp != fillable_.get()) {
      fillable_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}
