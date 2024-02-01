// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/DatetimeBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/util.h"

namespace awkward {
  const BuilderPtr
  DatetimeBuilder::fromempty(const BuilderOptions& options, const std::string& units) {
    GrowableBuffer<int64_t> content = GrowableBuffer<int64_t>::empty(options);
    return std::make_shared<DatetimeBuilder>(options,
                                             std::move(content),
                                             units);
  }

  DatetimeBuilder::DatetimeBuilder(const BuilderOptions& options,
                                   GrowableBuffer<int64_t> content,
                                   const std::string& units)
      : options_(options)
      , content_(std::move(content))
      , units_(units) { }

  const std::string
  DatetimeBuilder::classname() const {
    return "DatetimeBuilder";
  };

  const std::string
  DatetimeBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    content_.concatenate(
      reinterpret_cast<int64_t*>(
        container.empty_buffer(form_key.str() + "-data",
        (int64_t)content_.length() * (int64_t)sizeof(int64_t))));

    std::string primitive(units_);

    if (primitive.find("datetime64") == 0) {
      return "{\"class\": \"NumpyArray\", \"primitive\": \""
             + primitive + "\", \"format\": \""
             + "M8" + primitive.substr(10) + "\", \"form_key\": \""
             + form_key.str() + "\"}";
    }
    else if (primitive.find("timedelta64") == 0) {
      return "{\"class\": \"NumpyArray\", \"primitive\": \""
             + primitive + "\", \"format\": \""
             + "m8" + primitive.substr(11) + "\", \"form_key\": \""
             + form_key.str() + "\"}";
    }
    else {
      return "{\"class\": \"NumpyArray\", \"primitive\": \""
             + primitive + "\", \"form_key\": \""
             + form_key.str() + "\"}";
    }
  }

  int64_t
  DatetimeBuilder::length() const {
    return (int64_t)content_.length();
  }

  void
  DatetimeBuilder::clear() {
    content_.clear();
  }

  bool
  DatetimeBuilder::active() const {
    return false;
  }

  const BuilderPtr
  DatetimeBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::datetime(int64_t x, const std::string& unit) {
    if (unit == units_) {
      content_.append(x);
      return nullptr;
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->datetime(x, unit);
      return out;
    }
  }

  const BuilderPtr
  DatetimeBuilder::timedelta(int64_t x, const std::string& unit) {
    if (unit == units_) {
      content_.append(x);
      return nullptr;
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->timedelta(x, unit);
      return out;
    }
  }

  const BuilderPtr
  DatetimeBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::index(int64_t /* index */) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  void
  DatetimeBuilder::field(const char* /* key */, bool /* check */) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const std::string&
  DatetimeBuilder::units() const {
    return units_;
  }

}
