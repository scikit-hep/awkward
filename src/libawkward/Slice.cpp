// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/Slice.cpp", line)

#include <algorithm>
#include <sstream>
#include <type_traits>

#include "awkward/kernels/getitem.h"
#include "awkward/util.h"

#define AWKWARD_SLICE_NO_EXTERN_TEMPLATE
#include "awkward/Slice.h"

namespace awkward {
  ////////// SliceItem

  SliceItem::~SliceItem() = default;

  ////////// SliceAt

  SliceAt::SliceAt(int64_t at)
      : at_(at) { }

  int64_t
  SliceAt::at() const {
    return at_;
  }

  const SliceItemPtr
  SliceAt::shallow_copy() const {
    return std::make_shared<SliceAt>(at_);
  }

  const std::string
  SliceAt::tostring() const {
    return std::to_string(at_);
  }

  bool
  SliceAt::preserves_type(const Index64& advanced) const {
    return false;
  }

  ////////// SliceRange

  SliceRange::SliceRange(int64_t start, int64_t stop, int64_t step)
      : start_(start)
      , stop_(stop)
      , step_(step == Slice::none() ? 1 : step) {
    if (step_ == 0) {
      throw std::runtime_error(
        std::string("step must not be zero") + FILENAME(__LINE__));
    }
  }

  int64_t
  SliceRange::start() const {
    return start_;
  }

  int64_t
  SliceRange::stop() const {
    return stop_;
  }

  int64_t
  SliceRange::step() const {
    return step_;
  }

  bool
  SliceRange::hasstart() const {
    return start_ != Slice::none();
  }

  bool
  SliceRange::hasstop() const {
    return stop_ != Slice::none();
  }

  const SliceItemPtr
  SliceRange::shallow_copy() const {
    return std::make_shared<SliceRange>(start_, stop_, step_);
  }

  const std::string
  SliceRange::tostring() const {
    std::stringstream out;
    if (hasstart()) {
      out << start_;
    }
    out << ":";
    if (hasstop()) {
      out << stop_;
    }
    if (step_ != 1) {
      out << ":" << step_;
    }
    return out.str();
  }

  bool
  SliceRange::preserves_type(const Index64& advanced) const {
    return true;
  }

  ////////// SliceEllipsis

  SliceEllipsis::SliceEllipsis() { }

  const SliceItemPtr
  SliceEllipsis::shallow_copy() const {
    return std::make_shared<SliceEllipsis>();
  }

  const std::string
  SliceEllipsis::tostring() const {
    return std::string("...");
  }

  bool
  SliceEllipsis::preserves_type(const Index64& advanced) const {
    return true;
  }

  ////////// SliceNewAxis

  SliceNewAxis::SliceNewAxis() { }

  const SliceItemPtr
  SliceNewAxis::shallow_copy() const {
    return std::make_shared<SliceNewAxis>();
  }

  const std::string
  SliceNewAxis::tostring() const {
    return std::string("newaxis");
  }

  bool
  SliceNewAxis::preserves_type(const Index64& advanced) const {
    return false;
  }

  ////////// SliceArrayOf<T>

  template <typename T>
  SliceArrayOf<T>::SliceArrayOf(const IndexOf<T>& index,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                bool frombool)
      : index_(index)
      , shape_(shape)
      , strides_(strides)
      , frombool_(frombool) {
    if (shape_.empty()) {
      throw std::runtime_error(
        std::string("shape must not be zero-dimensional") + FILENAME(__LINE__));
    }
    if (shape_.size() != strides_.size()) {
      throw std::runtime_error(
        std::string("shape must have the same number of dimensions as strides")
        + FILENAME(__LINE__));
    }
  }

  template <typename T>
  const IndexOf<T>
  SliceArrayOf<T>::index() const {
    return index_;
  }

  template <typename T>
  const int64_t
  SliceArrayOf<T>::length() const {
    return shape_[0];
  }

  template <typename T>
  const std::vector<int64_t>
  SliceArrayOf<T>::shape() const {
    return shape_;
  }

  template <typename T>
  const std::vector<int64_t>
  SliceArrayOf<T>::strides() const {
    return strides_;
  }

  template <typename T>
  bool
  SliceArrayOf<T>::frombool() const {
    return frombool_;
  }

  template <typename T>
  int64_t
  SliceArrayOf<T>::ndim() const {
    return (int64_t)shape_.size();
  }

  template <typename T>
  const SliceItemPtr
  SliceArrayOf<T>::shallow_copy() const {
    return std::make_shared<SliceArrayOf<T>>(index_,
                                             shape_,
                                             strides_,
                                             frombool_);
  }

  template <typename T>
  const std::string
  SliceArrayOf<T>::tostring() const {
    return std::string("array(") + tostring_part() + std::string(")");
  }

  template <typename T>
  const std::string
  SliceArrayOf<T>::tostring_part() const {
    std::stringstream out;
    out << "[";
    if (shape_.size() == 1) {
      if (shape_[0] < 6) {
        for (int64_t i = 0;  i < shape_[0];  i++) {
          if (i != 0) {
            out << ", ";
          }
          out << (T)index_.getitem_at_nowrap(i*strides_[0]);
        }
      }
      else {
        for (int64_t i = 0;  i < 3;  i++) {
          if (i != 0) {
            out << ", ";
          }
          out << (T)index_.getitem_at_nowrap(i*strides_[0]);
        }
        out << ", ..., ";
        for (int64_t i = shape_[0] - 3;  i < shape_[0];  i++) {
          if (i != shape_[0] - 3) {
            out << ", ";
          }
          out << (T)index_.getitem_at_nowrap(i*strides_[0]);
        }
      }
    }
    else {
      std::vector<int64_t> shape(shape_.begin() + 1, shape_.end());
      std::vector<int64_t> strides(strides_.begin() + 1, strides_.end());
      if (shape_[0] < 6) {
        for (int64_t i = 0;  i < shape_[0];  i++) {
          if (i != 0) {
            out << ", ";
          }
          IndexOf<T> index(index_.ptr(),
                           index_.offset() + i*strides_[0],
                           shape_[1],
                           index_.ptr_lib());
          SliceArrayOf<T> subarray(index, shape, strides, frombool_);
          out << subarray.tostring_part();
        }
      }
      else {
        for (int64_t i = 0;  i < 3;  i++) {
          if (i != 0) {
            out << ", ";
          }
          IndexOf<T> index(index_.ptr(),
                           index_.offset() + i*strides_[0],
                           shape_[1],
                           index_.ptr_lib());
          SliceArrayOf<T> subarray(index, shape, strides, frombool_);
          out << subarray.tostring_part();
        }
        out << ", ..., ";
        for (int64_t i = shape_[0] - 3;  i < shape_[0];  i++) {
          if (i != shape_[0] - 3) {
            out << ", ";
          }
          IndexOf<T> index(index_.ptr(),
                           index_.offset() + i*strides_[0],
                           shape_[1],
                           index_.ptr_lib());
          SliceArrayOf<T> subarray(index, shape, strides, frombool_);
          out << subarray.tostring_part();
        }
      }
    }
    out << "]";
    return out.str();
  }

  template <typename T>
  bool
  SliceArrayOf<T>::preserves_type(const Index64& advanced) const {
    return advanced.length() == 0;
  }

  template <typename T>
  const IndexOf<T>
  SliceArrayOf<T>::ravel() const {
    int64_t length = 1;
    for (int64_t i = 0;  i < ndim();  i++) {
      length *= shape_[(size_t)i];
    }

    IndexOf<T> index(length);
    if (std::is_same<T, int64_t>::value) {
      struct Error err = kernel::slicearray_ravel_64(
        kernel::lib::cpu,   // DERIVE
        index.ptr().get(),
        index_.ptr().get(),
        ndim(),
        shape_.data(),
        strides_.data());
      util::handle_error(err);
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized SliceArrayOf<T> type")
        + FILENAME(__LINE__));
    }

    return index;
  }

  template class EXPORT_TEMPLATE_INST SliceArrayOf<int64_t>;

  ////////// SliceField

  SliceField::SliceField(const std::string& key)
      : key_(key) { }

  const std::string
  SliceField::key() const {
    return key_;
  }

  const SliceItemPtr
  SliceField::shallow_copy() const {
    return std::make_shared<SliceField>(key_);
  }

  const std::string
  SliceField::tostring() const {
    return util::quote(key_);
  }

  bool
  SliceField::preserves_type(const Index64& advanced) const {
    return false;
  }

  ////////// SliceFields

  SliceFields::SliceFields(const std::vector<std::string>& keys)
      : keys_(keys) { }

  const std::vector<std::string>
  SliceFields::keys() const {
    return keys_;
  }

  const SliceItemPtr
  SliceFields::shallow_copy() const {
    return std::make_shared<SliceFields>(keys_);
  }

  const std::string
  SliceFields::tostring() const {
    std::stringstream out;
    out << "[";
    for (size_t i = 0;  i < keys_.size();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << util::quote(keys_[i]);
    }
    out << "]";
    return out.str();
  }

  bool
  SliceFields::preserves_type(const Index64& advanced) const {
    return false;
  }

  ////////// SliceMissingOf<T>

  template <typename T>
  SliceMissingOf<T>::SliceMissingOf(const IndexOf<T>& index,
                                    const Index8& originalmask,
                                    const SliceItemPtr& content)
      : index_(index)
      , originalmask_(originalmask)
      , content_(content) { }

  template <typename T>
  int64_t
  SliceMissingOf<T>::length() const {
    return index_.length();
  }

  template <typename T>
  const IndexOf<T>
  SliceMissingOf<T>::index() const {
    return index_;
  }

  template <typename T>
  const Index8
  SliceMissingOf<T>::originalmask() const {
    return originalmask_;
  }

  template <typename T>
  const SliceItemPtr
  SliceMissingOf<T>::content() const {
    return content_;
  }

  template <typename T>
  const SliceItemPtr
  SliceMissingOf<T>::shallow_copy() const {
    return std::make_shared<SliceMissingOf<T>>(index_,
                                               originalmask_,
                                               content_);
  }

  template <typename T>
  const std::string
  SliceMissingOf<T>::tostring() const {
    return std::string("missing(") + tostring_part() + std::string(", ")
      + content_.get()->tostring() + std::string(")");
  }

  template <typename T>
  const std::string
  SliceMissingOf<T>::tostring_part() const {
    std::stringstream out;
    out << "[";
    if (index_.length() < 6) {
      for (int64_t i = 0;  i < index_.length();  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)index_.getitem_at_nowrap(i);
      }
    }
    else {
      for (int64_t i = 0;  i < 3;  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)index_.getitem_at_nowrap(i);
      }
      out << ", ..., ";
      for (int64_t i = index_.length() - 3;  i < index_.length();  i++) {
        if (i != index_.length() - 3) {
          out << ", ";
        }
        out << (T)index_.getitem_at_nowrap(i);
      }
    }
    out << "]";
    return out.str();
  }

  template <typename T>
  bool
  SliceMissingOf<T>::preserves_type(const Index64& advanced) const {
    return true;
  }

  template class EXPORT_TEMPLATE_INST SliceMissingOf<int64_t>;

  ////////// SliceJaggedOf<T>

  template <typename T>
  SliceJaggedOf<T>::SliceJaggedOf(const IndexOf<T>& offsets,
                                  const SliceItemPtr& content)
      : offsets_(offsets)
      , content_(content) { }

  template <typename T>
  int64_t
  SliceJaggedOf<T>::length() const {
    return offsets_.length() - 1;
  }

  template <typename T>
  const IndexOf<T>
  SliceJaggedOf<T>::offsets() const {
    return offsets_;
  }

  template <typename T>
  const SliceItemPtr
  SliceJaggedOf<T>::content() const {
    return content_;
  }

  template <typename T>
  const SliceItemPtr
  SliceJaggedOf<T>::shallow_copy() const {
    return std::make_shared<SliceJaggedOf<T>>(offsets_, content_);
  }

  template <typename T>
  const std::string
  SliceJaggedOf<T>::tostring() const {
    return std::string("jagged(") + tostring_part() + std::string(", ")
      + content_.get()->tostring() + std::string(")");
  }

  template <typename T>
  const std::string
  SliceJaggedOf<T>::tostring_part() const {
    std::stringstream out;
    out << "[";
    if (offsets_.length() < 6) {
      for (int64_t i = 0;  i < offsets_.length();  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)offsets_.getitem_at_nowrap(i);
      }
    }
    else {
      for (int64_t i = 0;  i < 3;  i++) {
        if (i != 0) {
          out << ", ";
        }
        out << (T)offsets_.getitem_at_nowrap(i);
      }
      out << ", ..., ";
      for (int64_t i = offsets_.length() - 3;  i < offsets_.length();  i++) {
        if (i != offsets_.length() - 3) {
          out << ", ";
        }
        out << (T)offsets_.getitem_at_nowrap(i);
      }
    }
    out << "]";
    return out.str();
  }

  template <typename T>
  bool
  SliceJaggedOf<T>::preserves_type(const Index64& advanced) const {
    return true;
  }

  template class EXPORT_TEMPLATE_INST SliceJaggedOf<int64_t>;

  ////////// Slice

  int64_t Slice::none() {
    return kSliceNone;
  }

  Slice::Slice()
      : items_(std::vector<SliceItemPtr>())
      , sealed_(false) { }

  Slice::Slice(const std::vector<SliceItemPtr>& items)
      : items_(items)
      , sealed_(false) { }

  Slice::Slice(const std::vector<SliceItemPtr>& items, bool sealed)
      : items_(items)
      , sealed_(sealed) { }

  const std::vector<SliceItemPtr>
  Slice::items() const {
    return items_;
  }

  bool
  Slice::sealed() const {
    return sealed_;
  }

  int64_t
  Slice::length() const {
    return (int64_t)items_.size();
  }

  int64_t
  Slice::dimlength() const {
    int64_t out = 0;
    for (auto x : items_) {
      if (dynamic_cast<SliceAt*>(x.get()) != nullptr) {
        out += 1;
      }
      else if (dynamic_cast<SliceRange*>(x.get()) != nullptr) {
        out += 1;
      }
      else if (dynamic_cast<SliceArray64*>(x.get()) != nullptr) {
        out += 1;
      }
    }
    return out;
  }

  const SliceItemPtr
  Slice::head() const {
    if (!items_.empty()) {
      return items_[0];
    }
    else {
      return SliceItemPtr(nullptr);
    }
  }

  const Slice
  Slice::tail() const {
    std::vector<SliceItemPtr> items;
    if (!items_.empty()) {
      items.insert(items.end(), items_.begin() + 1, items_.end());
    }
    return Slice(items, true);
  }

  const std::string
  Slice::tostring() const {
    std::stringstream out;
    out << "[";
    for (size_t i = 0;  i < items_.size();  i++) {
      if (i != 0) {
        out << ", ";
      }
      out << items_[i].get()->tostring();
    }
    out << "]";
    return out.str();
  }

  const Slice
  Slice::prepended(const SliceItemPtr& item) const {
    std::vector<SliceItemPtr> items(items_);
    items.insert(items.begin(), item);
    return Slice(items, true);
  }

  void
  Slice::append(const SliceItemPtr& item) {
    if (sealed_) {
      throw std::runtime_error(
        std::string("Slice::append when sealed_ == true")
        + FILENAME(__LINE__));
    }
    items_.push_back(item);
  }

  void
  Slice::append(const SliceAt& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceRange& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceEllipsis& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceNewAxis& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceArray64& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceField& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceFields& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceMissing64& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::append(const SliceJagged64& item) {
    items_.push_back(item.shallow_copy());
  }

  void
  Slice::become_sealed() {
    if (sealed_) {
      throw std::runtime_error(
        std::string("Slice::become_sealed when sealed_ == true")
        + FILENAME(__LINE__));
    }

    std::vector<int64_t> shape;
    for (size_t i = 0;  i < items_.size();  i++) {
      if (SliceArray64* array = dynamic_cast<SliceArray64*>(items_[i].get())) {
        if (shape.empty()) {
          shape = array->shape();
        }
        else if (shape.size() != (size_t)array->ndim()) {
          throw std::invalid_argument(
            std::string("cannot broadcast arrays in slice")
            + FILENAME(__LINE__));
        }
        else {
          std::vector<int64_t> arrayshape = array->shape();
          for (size_t j = 0;  j < shape.size();  j++) {
            if (arrayshape[j] == 0) {
              shape[j] = 0;
            }
            else if (arrayshape[j] > shape[j]) {
              shape[j] = arrayshape[j];
            }
          }
        }
      }
    }

    if (!shape.empty()) {
      for (size_t i = 0;  i < items_.size();  i++) {
        if (SliceAt* at = dynamic_cast<SliceAt*>(items_[i].get())) {
          Index64 index(1);
          index.setitem_at_nowrap(0, at->at());
          std::vector<int64_t> strides;
          for (size_t j = 0;  j < shape.size();  j++) {
            strides.push_back(0);
          }
          items_[i] = std::make_shared<SliceArray64>(index,
                                                     shape,
                                                     strides,
                                                     false);
        }
        else if (SliceArray64* array =
                 dynamic_cast<SliceArray64*>(items_[i].get())) {
          std::vector<int64_t> arrayshape = array->shape();
          std::vector<int64_t> arraystrides = array->strides();
          std::vector<int64_t> strides;
          for (size_t j = 0;  j < shape.size();  j++) {
            if (arrayshape[j] == shape[j]) {
              strides.push_back(arraystrides[j]);
            }
            else if (arrayshape[j] == 1) {
              strides.push_back(0);
            }
            else {
              throw std::invalid_argument(
                std::string("cannot broadcast arrays in slice")
                + FILENAME(__LINE__));
            }
          }
          items_[i] = std::make_shared<SliceArray64>(array->index(),
                                                     shape,
                                                     strides,
                                                     array->frombool());
        }
      }

      std::string types;
      for (size_t i = 0;  i < items_.size();  i++) {
        if (dynamic_cast<SliceAt*>(items_[i].get()) != nullptr) {
          types.push_back('@');
        }
        else if (dynamic_cast<SliceRange*>(items_[i].get()) != nullptr) {
          types.push_back(':');
        }
        else if (dynamic_cast<SliceEllipsis*>(items_[i].get()) != nullptr) {
          types.push_back('.');
        }
        else if (dynamic_cast<SliceNewAxis*>(items_[i].get()) != nullptr) {
          types.push_back('1');
        }
        else if (dynamic_cast<SliceArray64*>(items_[i].get()) != nullptr) {
          types.push_back('A');
        }
        else if (dynamic_cast<SliceField*>(items_[i].get()) != nullptr) {
          types.push_back('"');
        }
        else if (dynamic_cast<SliceFields*>(items_[i].get()) != nullptr) {
          types.push_back('[');
        }
        else if (dynamic_cast<SliceMissing64*>(items_[i].get()) != nullptr) {
          types.push_back('?');
        }
        else if (dynamic_cast<SliceJagged64*>(items_[i].get()) != nullptr) {
          types.push_back('J');
        }
      }

      if (std::count(types.begin(), types.end(), '.') > 1) {
        throw std::invalid_argument(
          std::string("a slice can have no more than one ellipsis ('...')")
          + FILENAME(__LINE__));
      }

      size_t numadvanced = (size_t)std::count(types.begin(), types.end(), 'A');
      if (numadvanced != 0) {
        types = types.substr(0, types.find_last_of('A') + 1)
                     .substr(types.find_first_of('A'));
        if (numadvanced != types.size()) {
          throw std::invalid_argument(
            std::string("advanced indexes separated by basic indexes is not permitted "
                        "(simple integers are advanced when any arrays are present)")
            + FILENAME(__LINE__));
        }
      }
    }

    sealed_ = true;
  }

  bool
  Slice::isadvanced() const {
    if (!sealed_) {
      throw std::runtime_error(
        std::string("Slice::isadvanced when sealed_ == false")
        + FILENAME(__LINE__));
    }
    for (size_t i = 0;  i < items_.size();  i++) {
      if (dynamic_cast<SliceArray64*>(items_[i].get()) != nullptr) {
        return true;
      }
    }
    return false;
  }
}
