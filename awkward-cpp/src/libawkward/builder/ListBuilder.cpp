// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ListBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/UnknownBuilder.h"

#include "awkward/builder/ListBuilder.h"

namespace awkward {
  const BuilderPtr
  ListBuilder::fromempty(const BuilderOptions& options) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    return std::make_shared<ListBuilder>(options,
                                         std::move(offsets),
                                         UnknownBuilder::fromempty(options),
                                         false);
  }

  ListBuilder::ListBuilder(const BuilderOptions& options,
                           GrowableBuffer<int64_t> offsets,
                           const BuilderPtr& content,
                           bool begun)
      : options_(options)
      , offsets_(std::move(offsets))
      , content_(content)
      , begun_(begun) { }

  const std::string
  ListBuilder::classname() const {
    return "ListBuilder";
  };

  const std::string
  ListBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    void* ptr = container.empty_buffer(form_key.str() + "-offsets",
      (int64_t)offsets_.length() * (int64_t)sizeof(int64_t));

    offsets_.concatenate(reinterpret_cast<int64_t*>(ptr));

    return "{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
           + content_.get()->to_buffers(container, form_key_id) + ", \"form_key\": \""
           + form_key.str() + "\"}";
  }

  int64_t
  ListBuilder::length() const {
    return (int64_t)offsets_.length() - 1;
  }

  void
  ListBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.get()->clear();
  }

  bool
  ListBuilder::active() const {
    return begun_;
  }

  const BuilderPtr
  ListBuilder::null() {
    if (!begun_) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
      out.get()->null();
      return out;
    }
    else {
      maybeupdate(content_.get()->null());
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::boolean(bool x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->boolean(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->boolean(x));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::integer(int64_t x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->integer(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->integer(x));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::real(double x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->real(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->real(x));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::complex(std::complex<double> x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->complex(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->complex(x));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::datetime(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->datetime(x, unit);
      return out;
    }
    else {
      maybeupdate(content_.get()->datetime(x, unit));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::timedelta(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->timedelta(x, unit);
      return out;
    }
    else {
      maybeupdate(content_.get()->timedelta(x, unit));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->string(x, length, encoding);
      return out;
    }
    else {
      maybeupdate(content_.get()->string(x, length, encoding));
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::beginlist() {
    if (!begun_) {
      begun_ = true;
    }
    else {
      maybeupdate(content_.get()->beginlist());
    }
    return shared_from_this();
  }

  const BuilderPtr
  ListBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (!content_.get()->active()) {
      offsets_.append(content_.get()->length());
      begun_ = false;
    }
    else {
      maybeupdate(content_.get()->endlist());
    }
    return shared_from_this();
  }

  const BuilderPtr
  ListBuilder::begintuple(int64_t numfields) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->begintuple(numfields);
      return out;
    }
    else {
      maybeupdate(content_.get()->begintuple(numfields));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->index(index);
      return nullptr;
    }
  }

  const BuilderPtr
  ListBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->endtuple();
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::beginrecord(const char* name, bool check) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->beginrecord(name, check);
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(name, check));
      return shared_from_this();
    }
  }

  void
  ListBuilder::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->field(key, check);
    }
  }

  const BuilderPtr
  ListBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else {
      content_.get()->endrecord();
      return shared_from_this();
    }
  }

  void
  ListBuilder::maybeupdate(const BuilderPtr tmp) {
    if (tmp  &&  tmp.get() != content_.get()) {
      content_ = std::move(tmp);
    }
  }
}
