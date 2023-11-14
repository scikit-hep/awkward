// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/UnionBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"
#include "awkward/builder/Complex128Builder.h"

#include "awkward/builder/UnionBuilder.h"

namespace awkward {
  const BuilderPtr
  UnionBuilder::fromsingle(const BuilderOptions& options,
                           const BuilderPtr& firstcontent) {
    std::vector<BuilderPtr> contents({ firstcontent });
    return std::make_shared<UnionBuilder>(options,
                                          GrowableBuffer<int8_t>::full(options, 0, firstcontent->length()),
                                          GrowableBuffer<int64_t>::arange(options, firstcontent->length()),
                                          contents);
  }

  UnionBuilder::UnionBuilder(const BuilderOptions& options,
                             GrowableBuffer<int8_t> tags,
                             GrowableBuffer<int64_t> index,
                             std::vector<BuilderPtr>& contents)
      : options_(options)
      , tags_(std::move(tags))
      , index_(std::move(index))
      , contents_(contents)
      , current_(-1) { }

  const std::string
  UnionBuilder::classname() const {
    return "UnionBuilder";
  };

  const std::string
  UnionBuilder::to_buffers(BuffersContainer& container, int64_t& form_key_id) const {
    std::stringstream form_key;
    form_key << "node" << (form_key_id++);

    tags_.concatenate(
      reinterpret_cast<int8_t*>(
        container.empty_buffer(form_key.str() + "-tags",
        (int64_t)tags_.length() * (int64_t)sizeof(int8_t))));

    index_.concatenate(
      reinterpret_cast<int64_t*>(
        container.empty_buffer(form_key.str() + "-index",
        (int64_t)index_.length() * (int64_t)sizeof(int64_t))));

    std::stringstream out;
    out << "{\"class\": \"UnionArray\", \"tags\": \"i8\", \"index\": \"i64\", \"contents\": [";
    for (size_t i = 0;  i < contents_.size();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << contents_[i].get()->to_buffers(container, form_key_id);
    }
    out << "], \"form_key\": \"" << form_key.str() + "\"}";
    return out.str();
  }

  int64_t
  UnionBuilder::length() const {
    return (int64_t)tags_.length();
  }

  void
  UnionBuilder::clear() {
    tags_.clear();
    index_.clear();
    for (auto x : contents_) {
      x.get()->clear();
    }
  }

  bool
  UnionBuilder::active() const {
    return current_ != -1;
  }

  const BuilderPtr
  UnionBuilder::null() {
    if (current_ == -1) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
      out.get()->null();
      return out;
    }
    else {
      contents_[(size_t)current_].get()->null();
      return shared_from_this();
    }
  }

  const BuilderPtr
  UnionBuilder::boolean(bool x) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        return dynamic_cast<BoolBuilder*>(p.get());
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(BoolBuilder::fromempty(options_));
        tofill = --contents_.end();
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t length = tofill->get()->length();
      tofill->get()->boolean(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->boolean(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::integer(int64_t x) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        return dynamic_cast<Int64Builder*>(p.get());
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(Int64Builder::fromempty(options_));
        tofill = --contents_.end();
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t len = tofill->get()->length();
      tofill->get()->integer(x);
      tags_.append(i);
      index_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->integer(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::real(double x) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        return dynamic_cast<Float64Builder*>(p.get());
      });
      if (tofill == contents_.end()) {
        tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
          return dynamic_cast<Int64Builder*>(p.get());
        });
        if (tofill != contents_.end()) {
          *tofill = Float64Builder::fromint64(
              options_,
              std::move(static_cast<Int64Builder*>(tofill->get())->buffer()));
        }
        else {
          contents_.emplace_back(Float64Builder::fromempty(options_));
          tofill = --contents_.end();
        }
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t length = tofill->get()->length();
      tofill->get()->real(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->real(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::complex(std::complex<double> x) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        return dynamic_cast<Complex128Builder*>(p.get());
      });
      if (tofill == contents_.end()) {
        tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
          return dynamic_cast<Float64Builder*>(p.get());
        });
        if (tofill != contents_.end()) {
          *tofill = std::move(Complex128Builder::fromfloat64(
              options_,
              std::move(static_cast<Float64Builder*>(tofill->get())->buffer())));
        }
      }
      if (tofill == contents_.end()) {
        tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
          return dynamic_cast<Int64Builder*>(p.get());
        });
        if (tofill != contents_.end()) {
          *tofill = std::move(Complex128Builder::fromint64(
              options_,
              std::move(static_cast<Int64Builder*>(tofill->get())->buffer())));
        }
        else {
          contents_.emplace_back(Complex128Builder::fromempty(options_));
          tofill = --contents_.end();
        }
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t length = tofill->get()->length();
      tofill->get()->complex(x);
      tags_.append(i);
      index_.append(length);
    }
    else {
      contents_[(size_t)current_].get()->complex(x);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::datetime(int64_t x, const std::string& unit) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        const auto& raw = dynamic_cast<DatetimeBuilder*>(p.get());
        return raw != 0 && raw->units() == unit;
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(DatetimeBuilder::fromempty(options_, unit));
        tofill = --contents_.end();
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t len = tofill->get()->length();
      tofill->get()->datetime(x, unit);
      tags_.append(i);
      index_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->datetime(x, unit);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::timedelta(int64_t x, const std::string& unit) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        const auto& raw = dynamic_cast<DatetimeBuilder*>(p.get());
        return raw != 0 && raw->units() == unit;
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(DatetimeBuilder::fromempty(options_, unit));
        tofill = --contents_.end();
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t len = tofill->get()->length();
      tofill->get()->timedelta(x, unit);
      tags_.append(i);
      index_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->timedelta(x, unit);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        const auto& raw = dynamic_cast<StringBuilder*>(p.get());
        return raw != 0 && raw->encoding() == encoding;
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(StringBuilder::fromempty(options_, encoding));
        tofill = --contents_.end();
      }
      int8_t i = (int8_t)std::distance(contents_.begin(), tofill);
      int64_t len = tofill->get()->length();
      tofill->get()->string(x, length, encoding);
      tags_.append(i);
      index_.append(len);
    }
    else {
      contents_[(size_t)current_].get()->string(x, length, encoding);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::beginlist() {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        return dynamic_cast<ListBuilder*>(p.get());
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(ListBuilder::fromempty(options_));
        tofill = --contents_.end();
      }
      tofill->get()->beginlist();
      current_ = (int8_t)std::distance(contents_.begin(), tofill);
    }
    else {
      contents_[(size_t)current_].get()->beginlist();
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::endlist() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endlist();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::begintuple(int64_t numfields) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        const auto& raw = dynamic_cast<TupleBuilder*>(p.get());
        return raw != nullptr  &&  (raw->length() == -1  ||  raw->numfields() == numfields);
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(TupleBuilder::fromempty(options_));
        tofill = --contents_.end();
      }
      tofill->get()->begintuple(numfields);
      current_ = (int8_t)std::distance(contents_.begin(), tofill);
    }
    else {
      contents_[(size_t)current_].get()->begintuple(numfields);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::index(int64_t index) {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)current_].get()->index(index);
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::endtuple() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endtuple();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return shared_from_this();
  }

  const BuilderPtr
  UnionBuilder::beginrecord(const char* name, bool check) {
    if (current_ == -1) {
      auto tofill = std::find_if(contents_.begin(), contents_.end(), [&](BuilderPtr const& p) {
        const auto& raw = dynamic_cast<RecordBuilder*>(p.get());
        return raw != nullptr  &&  (raw->length() == -1  ||
                ((check  &&  raw->name() == name)  ||
                 (!check  &&  raw->nameptr() == name)));
      });
      if (tofill == contents_.end()) {
        contents_.emplace_back(RecordBuilder::fromempty(options_));
        tofill = --contents_.end();
      }
      tofill->get()->beginrecord(name, check);
      current_ = (int8_t)std::distance(contents_.begin(), tofill);
    }
    else {
      contents_[(size_t)current_].get()->beginrecord(name, check);
    }
    return shared_from_this();
  }

  void
  UnionBuilder::field(const char* key, bool check) {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      contents_[(size_t)current_].get()->field(key, check);
    }
  }

  const BuilderPtr
  UnionBuilder::endrecord() {
    if (current_ == -1) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else {
      int64_t length = contents_[(size_t)current_].get()->length();
      contents_[(size_t)current_].get()->endrecord();
      if (length != contents_[(size_t)current_].get()->length()) {
        tags_.append(current_);
        index_.append(length);
        current_ = -1;
      }
    }
    return shared_from_this();
  }

}
