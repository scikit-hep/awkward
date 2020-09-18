// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/BoolBuilder.cpp", line)

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/BoolBuilder.h"

namespace awkward {
  const BuilderPtr
  BoolBuilder::fromempty(const ArrayBuilderOptions& options) {
    BuilderPtr out =
      std::make_shared<BoolBuilder>(options,
                                    GrowableBuffer<uint8_t>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  BoolBuilder::BoolBuilder(const ArrayBuilderOptions& options,
                           const GrowableBuffer<uint8_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string
  BoolBuilder::classname() const {
    return "BoolBuilder";
  };

  int64_t
  BoolBuilder::length() const {
    return buffer_.length();
  }

  void
  BoolBuilder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  BoolBuilder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::make_shared<NumpyArray>(Identities::none(),
                                        util::Parameters(),
                                        buffer_.ptr(),
                                        shape,
                                        strides,
                                        0,
                                        sizeof(bool),
                                        "?",
                                        util::dtype::boolean,
                                        kernel::lib::cpu);
  }

  bool
  BoolBuilder::active() const {
    return false;
  }

  const BuilderPtr
  BoolBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const BuilderPtr
  BoolBuilder::boolean(bool x) {
    buffer_.append(x);
    return that_;
  }

  const BuilderPtr
  BoolBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  BoolBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
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
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  BoolBuilder::index(int64_t index) {
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
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  BoolBuilder::field(const char* key, bool check) {
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

  const BuilderPtr
  BoolBuilder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }

}
