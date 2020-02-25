// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/ListType.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/UnionFillable.h"

#include "awkward/fillable/StringFillable.h"

namespace awkward {
  const std::shared_ptr<Fillable> StringFillable::fromempty(const FillableOptions& options, const char* encoding) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    GrowableBuffer<uint8_t> content = GrowableBuffer<uint8_t>::empty(options);
    std::shared_ptr<Fillable> out = std::make_shared<StringFillable>(options, offsets, content, encoding);
    out.get()->setthat(out);
    return out;
  }

  StringFillable::StringFillable(const FillableOptions& options, const GrowableBuffer<int64_t>& offsets, const GrowableBuffer<uint8_t>& content, const char* encoding)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , encoding_(encoding) { }

  const std::string StringFillable::classname() const {
    return "StringFillable";
  };

  const char* StringFillable::encoding() const {
    return encoding_;
  }

  int64_t StringFillable::length() const {
    return offsets_.length() - 1;
  }

  void StringFillable::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.clear();
  }

  const std::shared_ptr<Content> StringFillable::snapshot() const {
    util::Parameters char_parameters;
    char_parameters["__array__"] = std::string("\"char\"");

    util::Parameters string_parameters;
    string_parameters["__array__"] = std::string("\"string\"");

    if (encoding_ == nullptr) {
      char_parameters["__typestr__"] = std::string("\"byte\"");
      char_parameters["encoding"] = std::string("null");
      string_parameters["__typestr__"] = std::string("\"bytes\"");
    }
    else {
      std::string quoted = util::quote(encoding_, true);
      std::string slashquoted = std::string("\\\"") + quoted.substr(1, quoted.length() - 2) + std::string("\\\"");
      if (std::string(encoding_) == std::string("utf-8")) {
        char_parameters["__typestr__"] = std::string("\"utf8\"");
      string_parameters["__typestr__"] = std::string("\"string\"");
      }
      else {
        char_parameters["__typestr__"] = std::string("\"char[") + slashquoted + std::string("]\"");
        string_parameters["__typestr__"] = std::string("\"string[") + slashquoted + std::string("]\"");
      }
      char_parameters["encoding"] = std::string(quoted);
    }

    Index64 offsets(offsets_.ptr(), 0, offsets_.length());
    std::vector<ssize_t> shape = { (ssize_t)content_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(uint8_t) };
    std::shared_ptr<Content> content;
    content = std::make_shared<NumpyArray>(Identities::none(), char_parameters, content_.ptr(), shape, strides, 0, sizeof(uint8_t), "B");
    return std::make_shared<ListOffsetArray64>(Identities::none(), string_parameters, offsets, content);
  }

  bool StringFillable::active() const {
    return false;
  }

  const std::shared_ptr<Fillable> StringFillable::null() {
    std::shared_ptr<Fillable> out = OptionFillable::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::boolean(bool x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::integer(int64_t x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::real(double x) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::string(const char* x, int64_t length, const char* encoding) {
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

  const std::shared_ptr<Fillable> StringFillable::beginlist() {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::endlist() {
    throw std::invalid_argument("called 'endlist' without 'beginlist' at the same level before it");
  }

  const std::shared_ptr<Fillable> StringFillable::begintuple(int64_t numfields) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::index(int64_t index) {
    throw std::invalid_argument("called 'index' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> StringFillable::endtuple() {
    throw std::invalid_argument("called 'endtuple' without 'begintuple' at the same level before it");
  }

  const std::shared_ptr<Fillable> StringFillable::beginrecord(const char* name, bool check) {
    std::shared_ptr<Fillable> out = UnionFillable::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const std::shared_ptr<Fillable> StringFillable::field(const char* key, bool check) {
    throw std::invalid_argument("called 'field' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> StringFillable::endrecord() {
    throw std::invalid_argument("called 'endrecord' without 'beginrecord' at the same level before it");
  }

  const std::shared_ptr<Fillable> StringFillable::append(const std::shared_ptr<Content>& array, int64_t at) {
    throw std::runtime_error("FIXME: StringFillable::append");
  }
}
