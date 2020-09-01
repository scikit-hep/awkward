// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Int64Builder.cpp", line)

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/Float64Builder.h"

#include "awkward/builder/Int64Builder.h"

namespace awkward {
  const BuilderPtr
  Int64Builder::fromempty(const ArrayBuilderOptions& options) {
    BuilderPtr out =
      std::make_shared<Int64Builder>(options,
                                     GrowableBuffer<int64_t>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  Int64Builder::Int64Builder(const ArrayBuilderOptions& options,
                             const GrowableBuffer<int64_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const GrowableBuffer<int64_t>
  Int64Builder::buffer() const {
    return buffer_;
  }

  const std::string
  Int64Builder::classname() const {
    return "Int64Builder";
  };

  int64_t
  Int64Builder::length() const {
    return buffer_.length();
  }

  void
  Int64Builder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Int64Builder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
    return std::make_shared<NumpyArray>(
             Identities::none(),
             util::Parameters(),
             buffer_.ptr(),
             shape,
             strides,
             0,
             sizeof(int64_t),
             util::dtype_to_format(util::dtype::int64),
             util::dtype::int64,
             kernel::lib::cpu);
  }

  bool
  Int64Builder::active() const {
    return false;
  }

  const BuilderPtr
  Int64Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const BuilderPtr
  Int64Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  Int64Builder::integer(int64_t x) {
    buffer_.append(x);
    return that_;
  }

  const BuilderPtr
  Int64Builder::real(double x) {
    BuilderPtr out = Float64Builder::fromint64(options_, buffer_);
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  Int64Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  Int64Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
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
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  Int64Builder::index(int64_t index) {
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
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  Int64Builder::field(const char* key, bool check) {
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

  const BuilderPtr
  Int64Builder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
