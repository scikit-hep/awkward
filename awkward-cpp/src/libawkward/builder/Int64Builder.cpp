// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Int64Builder.cpp", line)

#include <stdexcept>

#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Int64Builder.h"

namespace awkward {
  const BuilderPtr
  Int64Builder::fromempty(const BuilderOptions& options) {
    return std::make_shared<Int64Builder>(options,
                                          GrowableBuffer<int64_t>::empty(options));
  }

  Int64Builder::Int64Builder(const BuilderOptions& options,
                             GrowableBuffer<int64_t> buffer)
      : options_(options)
      , buffer_(std::move(buffer)) { }

  GrowableBuffer<int64_t>
  Int64Builder::buffer() {
    // FIXME: swap with an empty buffer!
    return std::move(buffer_);
  }

  const std::string
  Int64Builder::classname() const {
    return "Int64Builder";
  };

  const std::string
  Int64Builder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    buffer_.concatenate(
      reinterpret_cast<int64_t*>(
        container.empty_buffer(form_key.str() + "-data",
        (int64_t)buffer_.length() * (int64_t)sizeof(int64_t))));

    return "{\"class\": \"NumpyArray\", \"primitive\": \"int64\", \"form_key\": \""
           + form_key.str() + "\"}";
  }

  int64_t
  Int64Builder::length() const {
    return (int64_t)buffer_.length();
  }

  void
  Int64Builder::clear() {
    buffer_.clear();
  }

  bool
  Int64Builder::active() const {
    return false;
  }

  const BuilderPtr
  Int64Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  Int64Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  Int64Builder::integer(int64_t x) {
    buffer_.append(x);
    return nullptr;
  }

  const BuilderPtr
  Int64Builder::real(double x) {
    BuilderPtr out = Float64Builder::fromint64(options_, buffer_);
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  Int64Builder::complex(std::complex<double> x) {
    BuilderPtr out = Complex128Builder::fromint64(options_, buffer_);
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  Int64Builder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  Int64Builder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  Int64Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  Int64Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  Int64Builder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Int64Builder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  Int64Builder::index(int64_t /* index */) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Int64Builder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Int64Builder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  void
  Int64Builder::field(const char* /* key */, bool /* check */) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Int64Builder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

}
