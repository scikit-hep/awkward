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
    return std::shared_ptr<Type>(new ArrayType(Type::Parameters(), fillable_.get()->type(), fillable_.get()->length()));
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

  const std::shared_ptr<Content> FillableArray::getitem_field(const std::string& key) const {
    return snapshot().get()->getitem_field(key);
  }

  const std::shared_ptr<Content> FillableArray::getitem_fields(const std::vector<std::string>& keys) const {
    return snapshot().get()->getitem_fields(keys);
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

  void FillableArray::bytestring(const char* x) {
    maybeupdate(fillable_.get()->string(x, -1, no_encoding));
  }

  void FillableArray::bytestring(const char* x, int64_t length) {
    maybeupdate(fillable_.get()->string(x, length, no_encoding));
  }

  void FillableArray::string(const char* x) {
    maybeupdate(fillable_.get()->string(x, -1, utf8_encoding));
  }

  void FillableArray::string(const char* x, int64_t length) {
    maybeupdate(fillable_.get()->string(x, length, utf8_encoding));
  }

  void FillableArray::beginlist() {
    maybeupdate(fillable_.get()->beginlist());
  }

  void FillableArray::endlist() {
    std::shared_ptr<Fillable> tmp = fillable_.get()->endlist();
    if (tmp == nullptr) {
      throw std::invalid_argument("endlist doesn't match a corresponding beginlist");
    }
    maybeupdate(tmp);
  }

  void FillableArray::begintuple(int64_t numfields) {
    maybeupdate(fillable_.get()->begintuple(numfields));
  }

  void FillableArray::index(int64_t index) {
    maybeupdate(fillable_.get()->index(index));
  }

  void FillableArray::endtuple() {
    maybeupdate(fillable_.get()->endtuple());
  }

  void FillableArray::beginrecord() {
    beginrecord_fast(nullptr);
  }

  void FillableArray::beginrecord_fast(const char* name) {
    maybeupdate(fillable_.get()->beginrecord(name, false));
  }

  void FillableArray::beginrecord_check(const char* name) {
    maybeupdate(fillable_.get()->beginrecord(name, true));
  }

  void FillableArray::field_fast(const char* key) {
    maybeupdate(fillable_.get()->field(key, false));
  }

  void FillableArray::field_check(const char* key) {
    maybeupdate(fillable_.get()->field(key, true));
  }

  void FillableArray::endrecord() {
    maybeupdate(fillable_.get()->endrecord());
  }

  void FillableArray::maybeupdate(const std::shared_ptr<Fillable>& tmp) {
    if (tmp.get() != fillable_.get()) {
      fillable_ = tmp;
    }
  }

  const char* FillableArray::no_encoding = nullptr;
  const char* FillableArray::utf8_encoding = "utf-8";
}

uint8_t awkward_FillableArray_length(void* fillablearray, int64_t* result) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    *result = obj->length();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_clear(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->clear();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_null(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->null();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_boolean(void* fillablearray, bool x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->boolean(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_integer(void* fillablearray, int64_t x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->integer(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_real(void* fillablearray, double x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->real(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_bytestring(void* fillablearray, const char* x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->bytestring(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_bytestring_length(void* fillablearray, const char* x, int64_t length) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->bytestring(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_string(void* fillablearray, const char* x) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->string(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_string_length(void* fillablearray, const char* x, int64_t length) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->string(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_beginlist(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->beginlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_endlist(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->endlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_begintuple(void* fillablearray, int64_t numfields) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->begintuple(numfields);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_index(void* fillablearray, int64_t index) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->index(index);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_endtuple(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->endtuple();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_beginrecord(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->beginrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_beginrecord_fast(void* fillablearray, const char* name) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->beginrecord_fast(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_beginrecord_check(void* fillablearray, const char* name) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->beginrecord_check(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_field_fast(void* fillablearray, const char* key) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->field_fast(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_field_check(void* fillablearray, const char* key) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->field_check(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_FillableArray_endrecord(void* fillablearray) {
  awkward::FillableArray* obj = reinterpret_cast<awkward::FillableArray*>(fillablearray);
  try {
    obj->endrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}
