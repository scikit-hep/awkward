// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/io/json.h"

namespace awkward {
  std::shared_ptr<Content> FromJsonString(const char* source) {
    throw std::runtime_error("FIXME");
  }

  std::shared_ptr<Content> FromJsonString(std::string source) {
    throw std::runtime_error("FIXME");
  }

  std::shared_ptr<Content> FromJsonFile(FILE* source) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::null() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::integer(int64_t x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::real(double x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::string(const char* x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::beginlist() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::endlist() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::beginrec() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::fieldname(const char* x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonString::endrec() {
    throw std::runtime_error("FIXME");
  }

  const char* ToJsonString::tocharstar() {
    throw std::runtime_error("FIXME");
  }

  std::string ToJsonString::tostring() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::null() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::integer(int64_t x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::real(double x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::string(const char* x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::beginlist() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::endlist() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::beginrec() {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::fieldname(const char* x) {
    throw std::runtime_error("FIXME");
  }

  void ToJsonFile::endrec() {
    throw std::runtime_error("FIXME");
  }

}
