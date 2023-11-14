// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/OptionBuilder.cpp", line)

#include <stdexcept>


#include "awkward/builder/OptionBuilder.h"

namespace awkward {
  const BuilderPtr
  OptionBuilder::fromnulls(const BuilderOptions& options,
                           int64_t nullcount,
                           const BuilderPtr& content) {
    return std::make_shared<OptionBuilder>(
      options,
      GrowableBuffer<int64_t>::full(options,
                                    -1,
                                    nullcount),
     content);
  }

  const BuilderPtr
  OptionBuilder::fromvalids(const BuilderOptions& options,
                            const BuilderPtr& content) {
    return std::make_shared<OptionBuilder>(options,
                                           GrowableBuffer<int64_t>::arange(options, content->length()),
                                           content);
  }

  OptionBuilder::OptionBuilder(const BuilderOptions&,
                               GrowableBuffer<int64_t> index,
                               const BuilderPtr content)
    : index_(std::move(index))
      , content_(content) { }

  const std::string
  OptionBuilder::classname() const {
    return "OptionBuilder";
  };

  const std::string
  OptionBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    void* ptr = container.empty_buffer(form_key.str() + "-index",
      (int64_t)index_.length() * (int64_t)sizeof(int64_t));

    index_.concatenate(reinterpret_cast<int64_t*>(ptr));

    return "{\"class\": \"IndexedOptionArray\", \"index\": \"i64\", \"content\": "
           + content_.get()->to_buffers(container, form_key_id) + ", \"form_key\": \""
           + form_key.str() + "\"}";
  }

  int64_t
  OptionBuilder::length() const {
    return (int64_t)index_.length();
  }

  void
  OptionBuilder::clear() {
    index_.clear();
    content_.get()->clear();
  }

  bool
  OptionBuilder::active() const {
    return content_.get()->active();
  }

  const BuilderPtr
  OptionBuilder::null() {
    if (!content_.get()->active()) {
      index_.append(-1);
    }
    else {
      content_.get()->null();
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::boolean(bool x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->boolean(x));
      index_.append(length);
    }
    else {
      content_.get()->boolean(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::integer(int64_t x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->integer(x));
      index_.append(length);
    }
    else {
      content_.get()->integer(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::real(double x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->real(x));
      index_.append(length);
    }
    else {
      content_.get()->real(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::complex(std::complex<double> x) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->complex(x));
      index_.append(length);
    }
    else {
      content_.get()->complex(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::datetime(int64_t x, const std::string& unit) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->datetime(x, unit));
      index_.append(length);
    }
    else {
      content_.get()->datetime(x, unit);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::timedelta(int64_t x, const std::string& unit) {
    if (!content_.get()->active()) {
      int64_t length = content_.get()->length();
      maybeupdate(content_.get()->timedelta(x, unit));
      index_.append(length);
    }
    else {
      content_.get()->timedelta(x, unit);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!content_.get()->active()) {
      int64_t len = content_.get()->length();
      maybeupdate(content_.get()->string(x, length, encoding));
      index_.append(len);
    }
    else {
      content_.get()->string(x, length, encoding);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::beginlist() {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginlist());
    }
    else {
      content_.get()->beginlist();
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::endlist() {
    if (!content_.get()->active()) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endlist();
      if (length != content_.get()->length()) {
        index_.append(length);
      }
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::begintuple(int64_t numfields) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->begintuple(numfields));
    }
    else {
      content_.get()->begintuple(numfields);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::index(int64_t index) {
    if (!content_.get()->active()) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->index(index);
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::endtuple() {
    if (!content_.get()->active()) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endtuple();
      if (length != content_.get()->length()) {
        index_.append(length);
      }
    }
    return shared_from_this();
  }

  const BuilderPtr
  OptionBuilder::beginrecord(const char* name, bool check) {
    if (!content_.get()->active()) {
      maybeupdate(content_.get()->beginrecord(name, check));
    }
    else {
      content_.get()->beginrecord(name, check);
    }
    return shared_from_this();
  }

  void
  OptionBuilder::field(const char* key, bool check) {
    if (!content_.get()->active()) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->field(key, check);
    }
  }

  const BuilderPtr
  OptionBuilder::endrecord() {
    if (!content_.get()->active()) {
      throw std::invalid_argument(
        std::string("called 'endrecord' without 'beginrecord' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else {
      int64_t length = content_.get()->length();
      content_.get()->endrecord();
      if (length != content_.get()->length()) {
        index_.append(length);
      }
    }
    return shared_from_this();
  }

  void
  OptionBuilder::maybeupdate(const BuilderPtr tmp) {
    if (tmp  &&  tmp.get() != content_.get()) {
      content_ = std::move(tmp);
    }
  }
}
