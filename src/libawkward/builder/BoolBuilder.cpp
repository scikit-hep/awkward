// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/BoolBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> BoolBuilder::fromempty(const ArrayBuilderOptions& options) {
    std::shared_ptr<Builder> out = std::make_shared<BoolBuilder>(options, GrowableBuffer<uint8_t>::empty(options));
    out.get()->setthat(out);
    return out;
  }

  BoolBuilder::BoolBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<uint8_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string BoolBuilder::classname() const {
    return "BoolBuilder";
  };

  int64_t BoolBuilder::length() const {
    return buffer_.length();
  }

  void BoolBuilder::clear() {
    buffer_.clear();
  }

  ContentPtr BoolBuilder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::make_shared<NumpyArray>(Identities::none(), util::Parameters(), buffer_.ptr(), shape, strides, 0, sizeof(bool), "?");
  }

  bool BoolBuilder::active() const {
    return false;
  }

  const std::shared_ptr<Builder> BoolBuilder::null() {
    std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::boolean(bool x) {
    buffer_.append(x);
    return that_;
  }

  const std::shared_ptr<Builder> BoolBuilder::integer(int64_t x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::real(double x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::string(const char* x, int64_t length, const char* encoding) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->string(x, length, encoding);
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::beginlist() {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Builder> BoolBuilder::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> BoolBuilder::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> BoolBuilder::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Builder> BoolBuilder::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> BoolBuilder::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> BoolBuilder::append(ContentPtr& array, int64_t at) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }

}
