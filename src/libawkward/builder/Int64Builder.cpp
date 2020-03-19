// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/Float64Builder.h"

#include "awkward/builder/Int64Builder.h"

namespace awkward {
  const std::shared_ptr<Builder> Int64Builder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<Int64Builder>(options, GrowableBuffer<int64_t>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  Int64Builder::Int64Builder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const GrowableBuffer<int64_t> Int64Builder::buffer() const {
    return buffer_;
  }

  const std::string Int64Builder::classname() const {
    return "Int64Builder";
  };

  int64_t Int64Builder::length() const {
    return buffer_.length();
  }

  void Int64Builder::clear() {
    buffer_.clear();
  }

  ContentPtr Int64Builder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
#if defined _MSC_VER || defined __i386__
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "q");
#else
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), buffer_.ptr(), shape, strides, 0, sizeof(int64_t), "l");
#endif
  }

  bool Int64Builder::active() const {
    return false;
  }

  const std::shared_ptr<Builder> Int64Builder::null() {
    std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::boolean(bool x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::integer(int64_t x) {
    buffer_.append(x);
    return that_;
  }

  const std::shared_ptr<Builder> Int64Builder::real(double x) {
    std::shared_ptr<Builder> out = Float64Builder::fromint64(options_, buffer_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::beginlist() {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Builder> Int64Builder::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> Int64Builder::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> Int64Builder::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Builder> Int64Builder::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> Int64Builder::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> Int64Builder::append(ContentPtr& array, int64_t at) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
