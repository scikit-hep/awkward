// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Float64Builder.cpp", line)

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Float64Builder.h"

namespace awkward {
  const BuilderPtr
  Float64Builder::fromempty(const ArrayBuilderOptions& options) {
    BuilderPtr out =
      std::make_shared<Float64Builder>(options,
                                       GrowableBuffer<double>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  const BuilderPtr
  Float64Builder::fromint64(const ArrayBuilderOptions& options,
                            const GrowableBuffer<int64_t>& old) {
    GrowableBuffer<double> buffer =
      GrowableBuffer<double>::empty(options, old.reserved());
    int64_t* oldraw = old.ptr().get();
    double* newraw = buffer.ptr().get();
    for (int64_t i = 0;  i < old.length();  i++) {
      newraw[i] = (double)oldraw[i];
    }
    buffer.set_length(old.length());
    BuilderPtr out = std::make_shared<Float64Builder>(options, buffer);
    out.get()->setthat(out);
    return out;
  }

  Float64Builder::Float64Builder(const ArrayBuilderOptions& options,
                                 const GrowableBuffer<double>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string
  Float64Builder::classname() const {
    return "Float64Builder";
  }

  int64_t
  Float64Builder::length() const {
    return buffer_.length();
  }

  void
  Float64Builder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Float64Builder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
    return std::make_shared<NumpyArray>(Identities::none(),
                                        util::Parameters(),
                                        buffer_.ptr(),
                                        shape,
                                        strides,
                                        0,
                                        sizeof(double),
                                        "d",
                                        util::dtype::float64,
                                        kernel::lib::cpu);
  }

  bool
  Float64Builder::active() const {
    return false;
  }

  const BuilderPtr
  Float64Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const BuilderPtr
  Float64Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  Float64Builder::integer(int64_t x) {
    buffer_.append((double)x);
    return that_;
  }

  const BuilderPtr
  Float64Builder::real(double x) {
    buffer_.append(x);
    return that_;
  }

  const BuilderPtr
  Float64Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  Float64Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  Float64Builder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  Float64Builder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  Float64Builder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
