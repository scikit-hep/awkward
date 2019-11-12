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

  void FillableArray::integer(int64_t x) {
    maybeupdate(fillable_.get()->integer(x));
  }

  void FillableArray::real(double x) {
    maybeupdate(fillable_.get()->real(x));
  }

  void FillableArray::beginlist() {
    maybeupdate(fillable_.get()->beginlist());
  }

  void FillableArray::endlist() {
    Fillable* tmp = fillable_.get()->endlist();
    if (tmp == nullptr) {
      throw std::invalid_argument("endlist doesn't match a corresponding beginlist");
    }
    maybeupdate(tmp);
  }

  void FillableArray::maybeupdate(Fillable* tmp) {
    if (tmp != fillable_.get()  &&  tmp != nullptr) {
      fillable_ = std::shared_ptr<Fillable>(tmp);
    }
  }
}

bool awkward_FillableArray_length(void* fillablearray, int64_t& result) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    result = obj->length();
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_clear(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->clear();
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_null(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->null();
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_boolean(void* fillablearray, bool x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->boolean(x);
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_integer(void* fillablearray, int64_t x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->integer(x);
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_real(void* fillablearray, double x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->real(x);
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_beginlist(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->beginlist();
  }
  catch (...) {
    return false;
  }
  return true;
}

bool awkward_FillableArray_endlist(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->endlist();
  }
  catch (...) {
    return false;
  }
  return true;
}
