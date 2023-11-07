// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/BoolBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/BoolBuilder.h"

namespace awkward {
  const BuilderPtr
  BoolBuilder::fromempty(const BuilderOptions& options) {
    return std::make_shared<BoolBuilder>(options,
                                         std::move(GrowableBuffer<uint8_t>::empty(options)));
  }

  BoolBuilder::BoolBuilder(const BuilderOptions& options,
                           GrowableBuffer<uint8_t> buffer)
      : options_(options)
      , buffer_(std::move(buffer)) { }

  const std::string
  BoolBuilder::classname() const {
    return "BoolBuilder";
  };

  const std::string
  BoolBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    buffer_.concatenate(
      reinterpret_cast<uint8_t*>(
        container.empty_buffer(form_key.str() + "-data",
        (int64_t)buffer_.length() * (int64_t)sizeof(bool))));

    return "{\"class\": \"NumpyArray\", \"primitive\": \"bool\", \"form_key\": \""
           + form_key.str() + "\"}";
  }

  int64_t
  BoolBuilder::length() const {
    return (int64_t)buffer_.length();
  }

  void
  BoolBuilder::clear() {
    buffer_.clear();
  }

  bool
  BoolBuilder::active() const {
    return false;
  }

  const BuilderPtr
  BoolBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  BoolBuilder::boolean(bool x) {
    buffer_.append(x);
    return nullptr;
  }

  const BuilderPtr
  BoolBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  BoolBuilder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  BoolBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  BoolBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  BoolBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  BoolBuilder::index(int64_t /* index */) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begintuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'endtuple' without 'begintuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  void
  BoolBuilder::field(const char* /* key */, bool /* check */) {
    throw std::invalid_argument(
      std::string("called 'field' without 'beginrecord' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'endrecord' without 'beginrecord' at the same level before it")
      + FILENAME(__LINE__));
  }

}
