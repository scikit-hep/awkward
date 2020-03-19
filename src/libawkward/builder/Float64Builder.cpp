// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Float64Builder.h"

namespace awkward {
  const std::shared_ptr<Builder> Float64Builder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<Float64Builder>(options, GrowableBuffer<double>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::fromint64(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& old) {
    GrowableBuffer<double> buffer = GrowableBuffer<double>::empty(options, old.reserved());
    int64_t* oldraw = old.ptr().get();
    double* newraw = buffer.ptr().get();
    for (int64_t i = 0;  i < old.length();  i++) {
      newraw[i] = (double)oldraw[i];
    }
    buffer.set_length(old.length());
    std::shared_ptr<Builder> out = std::make_shared<Float64Builder>(options, buffer);
    out.get()->setthat(out);
    return out;
  }

  Float64Builder::Float64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<double>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string Float64Builder::classname() const {
    return "Float64Builder";
  }

  int64_t Float64Builder::length() const {
    return buffer_.length();
  }

  void Float64Builder::clear() {
    buffer_.clear();
  }

  const ContentPtr Float64Builder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), buffer_.ptr(), shape, strides, 0, sizeof(double), "d");
  }

  bool Float64Builder::active() const {
    return false;
  }

  const std::shared_ptr<Builder> Float64Builder::null() {
    std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::boolean(bool x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::integer(int64_t x) {
    buffer_.append((double)x);
    return that_;
  }

  const std::shared_ptr<Builder> Float64Builder::real(double x) {
    buffer_.append(x);
    return that_;
  }

  const std::shared_ptr<Builder> Float64Builder::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::beginlist() {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Builder> Float64Builder::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> Float64Builder::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> Float64Builder::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Builder> Float64Builder::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> Float64Builder::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> Float64Builder::append(const ContentPtr& array, int64_t at) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
