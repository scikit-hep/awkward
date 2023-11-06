// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ArrayBuilder.cpp", line)

#include <sstream>
#include <stdexcept>

#include "awkward/common.h"
#include "awkward/builder/Builder.h"
#include "awkward/builder/UnknownBuilder.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/BuilderOptions.h"

namespace awkward {
  ArrayBuilder::ArrayBuilder(const BuilderOptions& options)
      : builder_(UnknownBuilder::fromempty(options)) { }

  const std::string
  ArrayBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    return builder_.get()->to_buffers(container, form_key_id);
  }

  int64_t
  ArrayBuilder::length() const {
    return builder_.get()->length();
  }

  void
  ArrayBuilder::clear() {
    if (builder_) {
      builder_.get()->clear();
    }
  }

  void
  ArrayBuilder::null() {
    maybeupdate(builder_.get()->null());
  }

  void
  ArrayBuilder::boolean(bool x) {
    maybeupdate(builder_.get()->boolean(x));
  }

  void
  ArrayBuilder::integer(int64_t x) {
    maybeupdate(builder_.get()->integer(x));
  }

  void
  ArrayBuilder::real(double x) {
    maybeupdate(builder_.get()->real(x));
  }

  void
  ArrayBuilder::complex(std::complex<double> x) {
    maybeupdate(builder_.get()->complex(x));
  }

  void
  ArrayBuilder::datetime(int64_t x, const std::string& unit) {
    maybeupdate(builder_.get()->datetime(x, unit));
  }

  void
  ArrayBuilder::timedelta(int64_t x, const std::string& unit) {
    maybeupdate(builder_.get()->timedelta(x, unit));
  }

  void
  ArrayBuilder::bytestring(const char* x) {
    maybeupdate(builder_.get()->string(x, -1, no_encoding));
  }

  void
  ArrayBuilder::bytestring(const char* x, int64_t length) {
    maybeupdate(builder_.get()->string(x, length, no_encoding));
  }

  void
  ArrayBuilder::bytestring(const std::string& x) {
    bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  ArrayBuilder::string(const char* x) {
    maybeupdate(builder_.get()->string(x, -1, utf8_encoding));
  }

  void
  ArrayBuilder::string(const char* x, int64_t length) {
    maybeupdate(builder_.get()->string(x, length, utf8_encoding));
  }

  void
  ArrayBuilder::string(const std::string& x) {
    string(x.c_str(), (int64_t)x.length());
  }

  void
  ArrayBuilder::beginlist() {
    maybeupdate(builder_.get()->beginlist());
  }

  void
  ArrayBuilder::endlist() {
    BuilderPtr tmp = builder_.get()->endlist();
    if (tmp.get() == nullptr) {
      throw std::invalid_argument(
        std::string("endlist doesn't match a corresponding beginlist")
        + FILENAME(__LINE__));
    }
    maybeupdate(tmp);
  }

  void
  ArrayBuilder::begintuple(int64_t numfields) {
    maybeupdate(builder_.get()->begintuple(numfields));
  }

  void
  ArrayBuilder::index(int64_t index) {
    maybeupdate(builder_.get()->index(index));
  }

  void
  ArrayBuilder::endtuple() {
    maybeupdate(builder_.get()->endtuple());
  }

  void
  ArrayBuilder::beginrecord() {
    beginrecord_fast(nullptr);
  }

  void
  ArrayBuilder::beginrecord_fast(const char* name) {
    maybeupdate(builder_.get()->beginrecord(name, false));
  }

  void
  ArrayBuilder::beginrecord_check(const char* name) {
    maybeupdate(builder_.get()->beginrecord(name, true));
  }

  void
  ArrayBuilder::beginrecord_check(const std::string& name) {
    beginrecord_check(name.c_str());
  }

  void
  ArrayBuilder::field_fast(const char* key) {
    builder_.get()->field(key, false);
  }

  void
  ArrayBuilder::field_check(const char* key) {
    builder_.get()->field(key, true);
  }

  void
  ArrayBuilder::field_check(const std::string& key) {
    field_check(key.c_str());
  }

  void
  ArrayBuilder::endrecord() {
    maybeupdate(builder_.get()->endrecord());
  }

  void
  ArrayBuilder::maybeupdate(const BuilderPtr tmp) {
    if (tmp  &&  tmp.get() != builder_.get()) {
      builder_ = std::move(tmp);
    }
  }

  const char* ArrayBuilder::no_encoding = nullptr;
  const char* ArrayBuilder::utf8_encoding = "utf-8";
}

////////// extern C interface

uint8_t awkward_ArrayBuilder_length(void* arraybuilder,
                                    int64_t* result) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    *result = obj->length();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_clear(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->clear();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_null(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->null();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_boolean(void* arraybuilder,
                                     bool x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->boolean(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_integer(void* arraybuilder,
                                     int64_t x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->integer(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_real(void* arraybuilder,
                                  double x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->real(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_complex(void* arraybuilder,
                                     double real,
                                     double imag) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->complex(std::complex<double>(real, imag));
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_datetime(void* arraybuilder,
                                      int64_t x,
                                      const char* unit) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  std::string unit_str(unit);
  try {
    obj->datetime(x, unit_str);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_timedelta(void* arraybuilder,
                                       int64_t x,
                                       const char* unit) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  std::string unit_str(unit);
  try {
    obj->timedelta(x, unit_str);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_bytestring(void* arraybuilder,
                                        const char* x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->bytestring(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_bytestring_length(void* arraybuilder,
                                               const char* x,
                                               int64_t length) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->bytestring(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_string(void* arraybuilder,
                                    const char* x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->string(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_string_length(void* arraybuilder,
                                           const char* x,
                                           int64_t length) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->string(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginlist(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endlist(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_begintuple(void* arraybuilder,
                                        int64_t numfields) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->begintuple(numfields);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_index(void* arraybuilder,
                                   int64_t index) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->index(index);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endtuple(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endtuple();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord_fast(void* arraybuilder,
                                              const char* name) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord_fast(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord_check(void* arraybuilder,
                                               const char* name) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord_check(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_field_fast(void* arraybuilder,
                                        const char* key) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->field_fast(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_field_check(void* arraybuilder,
                                         const char* key) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->field_check(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endrecord(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}
