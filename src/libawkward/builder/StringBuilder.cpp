// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/StringBuilder.cpp", line)

#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/type/PrimitiveType.h"
#include "awkward/type/ListType.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/StringBuilder.h"

namespace awkward {
  const BuilderPtr
  StringBuilder::fromempty(const ArrayBuilderOptions& options,
                           const char* encoding) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    GrowableBuffer<uint8_t> content = GrowableBuffer<uint8_t>::empty(options);
    BuilderPtr out = std::make_shared<StringBuilder>(options,
                                                     offsets,
                                                     content,
                                                     encoding);
    out.get()->setthat(out);
    return out;
  }

  StringBuilder::StringBuilder(const ArrayBuilderOptions& options,
                               const GrowableBuffer<int64_t>& offsets,
                               const GrowableBuffer<uint8_t>& content,
                               const char* encoding)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , encoding_(encoding) { }

  const std::string
  StringBuilder::classname() const {
    return "StringBuilder";
  };

  const char*
  StringBuilder::encoding() const {
    return encoding_;
  }

  int64_t
  StringBuilder::length() const {
    return offsets_.length() - 1;
  }

  void
  StringBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.clear();
  }

  const ContentPtr
  StringBuilder::snapshot() const {
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
      throw std::invalid_argument(
        std::string("unsupported encoding: ") + util::quote(encoding_)
        + FILENAME(__LINE__));
    }

    Index64 offsets(offsets_.ptr(), 0, offsets_.length(), kernel::lib::cpu);
    std::vector<ssize_t> shape = { (ssize_t)content_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(uint8_t) };
    ContentPtr content;
    content = std::make_shared<NumpyArray>(Identities::none(),
                                           char_parameters,
                                           content_.ptr(),
                                           shape,
                                           strides,
                                           0,
                                           sizeof(uint8_t),
                                           "B",
                                           util::dtype::uint8,
                                           kernel::lib::cpu);
    return std::make_shared<ListOffsetArray64>(Identities::none(),
                                               string_parameters,
                                               offsets,
                                               content);
  }

  bool
  StringBuilder::active() const {
    return false;
  }

  const BuilderPtr
  StringBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, that_);
    out.get()->null();
    return out;
  }

  const BuilderPtr
  StringBuilder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  StringBuilder::string(const char* x, int64_t length, const char* encoding) {
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

  const BuilderPtr
  StringBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  StringBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  StringBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  StringBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  StringBuilder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, that_);
    out.get()->append(array, at);
    return out;
  }
}
