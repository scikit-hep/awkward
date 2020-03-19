// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/ListType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/StringBuilder.h"

namespace awkward {
  const std::shared_ptr<Builder> StringBuilder::fromempty(const ArrayBuilderOptions& options, const char* encoding) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    GrowableBuffer<uint8_t> content = GrowableBuffer<uint8_t>::empty(options);
    std::shared_ptr<Builder> out = std::make_shared<StringBuilder>(options, offsets, content, encoding);
    out.get()->setthat(out);
    return out;
  }

  StringBuilder::StringBuilder(const ArrayBuilderOptions& options, const GrowableBuffer<int64_t>& offsets, const GrowableBuffer<uint8_t>& content, const char* encoding)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , encoding_(encoding) { }

  const std::string StringBuilder::classname() const {
    return "StringBuilder";
  };

  const char* StringBuilder::encoding() const {
    return encoding_;
  }

  int64_t StringBuilder::length() const {
    return offsets_.length() - 1;
  }

  void StringBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.clear();
  }

  ContentPtr StringBuilder::snapshot() const {
    util::Parameters char_parameters;
    util::Parameters string_parameters;

    if (encoding_ == nullptr) {
      char_parameters["__array__"] = std::string("\"byte\"");
      string_parameters["__array__"] = std::string("\"bytestring\"");
    }
    else if (std::string(encoding_) == std::string("utf-8")) {
      char_parameters["__array__"] = std::string("\"char\"");
      string_parameters["__array__"] = std::string("\"string\"");
    }
    else {
      throw std::invalid_argument(std::string("unsupported encoding: ") + util::quote(encoding_, false));
    }

    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    std::vector<ssize_t> shape = { (ssize_t)content_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(uint8_t) };
    std::shared_ptr<Content> content;
    content = std::make_shared<NumpyArray>(Identities::none(), char_parameters, content_.ptr(), shape, strides, 0, sizeof(uint8_t), "B");
    return std::make_shared<ListOffsetArray64>(Identities::none(), string_parameters, offsets, content);
  }

  bool StringBuilder::active() const {
    return false;
  }

  const std::shared_ptr<Builder> StringBuilder::null() {
    std::shared_ptr<Builder> out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::boolean(bool x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::integer(int64_t x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::real(double x) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (length < 0) {
      for (int64_t i = 0;  x[i] != 0;  i++) {
        content_.append((uint8_t)x[i]);
      }
    }
    else {
      for (int64_t i = 0;  i < length;  i++) {
        content_.append((uint8_t)x[i]);
      }
    }
    offsets_.append(content_.length());
    return that_;
  }

  const std::shared_ptr<Builder> StringBuilder::beginlist() {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Builder> StringBuilder::begintuple(int64_t numfields) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> StringBuilder::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Builder> StringBuilder::beginrecord(const char* name, bool check) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Builder> StringBuilder::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> StringBuilder::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Builder> StringBuilder::append(ContentPtr& array, int64_t at) {
    std::shared_ptr<Builder> out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
