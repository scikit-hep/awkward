// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/builder/TypedBuilder.h"
#include "awkward/Identities.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {

  template<typename T>
  TypedBuilder<T>::~TypedBuilder<T>() = default;

  /// BoolTypedBuilder
  ///
  BoolTypedBuilder::BoolTypedBuilder(const ArrayBuilderOptions& options,
                                     const GrowableBuffer<uint8_t>& buffer)
    : options_(options),
      buffer_(buffer) { }

  const std::string
  BoolTypedBuilder::classname() const {
    return "BoolTypedBuilder";
  }

  int64_t
  BoolTypedBuilder::length() const {
    return buffer_.length();
  }

  void
  BoolTypedBuilder::clear(){
    buffer_.clear();
  }

  const ContentPtr
  BoolTypedBuilder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
    return std::make_shared<NumpyArray>(Identities::none(),
                                        util::Parameters(),
                                        buffer_.ptr(),
                                        shape,
                                        strides,
                                        0,
                                        sizeof(bool),
                                        "?",
                                        util::dtype::boolean,
                                        kernel::lib::cpu);
  }


  /// @copydoc Builder::active()
  ///
  /// A BoolTypedBuilder is never active.
  bool
  BoolTypedBuilder::active() const {
    return false;
  }

  void
  BoolTypedBuilder::boolean(bool x) {
    buffer_.append(x);
    //return std::string("added bool ") + std::string(x ? "true" : "false");
  }

  Float64TypedBuilder::Float64TypedBuilder(const ArrayBuilderOptions& options,
                                           const GrowableBuffer<double>& buffer)
    : options_(options),
      buffer_(buffer) { }

  const std::string
  Float64TypedBuilder::classname() const {
    return "Float64TypedBuilder";
  }

  int64_t
  Float64TypedBuilder::length() const {
    return buffer_.length();
  }

  void
  Float64TypedBuilder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Float64TypedBuilder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
    return std::make_shared<NumpyArray>(Identities::none(),
                                        util::Parameters(),
                                        buffer_.ptr(),
                                        shape,
                                        strides,
                                        0,
                                        sizeof(double),
                                        "d",
                                        util::dtype::float64,
                                        kernel::lib::cpu);
  }

  /// @copydoc Builder::active()
  ///
  /// A Float64TypedBuilder is never active.
  bool
  Float64TypedBuilder::active() const {
    return false;
  }

  void
  Float64TypedBuilder::real(double x) {
    buffer_.append(x);
    //return std::string("added real ") + std::to_string(x);
  }

  template <typename T>
  IndexedTypedBuilder<T>::IndexedTypedBuilder(const ArrayBuilderOptions& options,
                                              const GrowableBuffer<int64_t>& index,
                                              const std::shared_ptr<T>& array,
                                              bool hasnull)
      : options_(options)
      , index_(index)
      , array_(array)
      , hasnull_(hasnull) { }

  //
  template <typename T>
  const Content*
  IndexedTypedBuilder<T>::arrayptr() const {
    return array_.get();
  }

  template <typename T>
  int64_t
  IndexedTypedBuilder<T>::length() const {
    return index_.length();
  }

  template <typename T>
  void
  IndexedTypedBuilder<T>::clear() {
    index_.clear();
  }

  template <typename T>
  bool
  IndexedTypedBuilder<T>::active() const {
    return false;
  }

  template <typename T>
  void
  IndexedTypedBuilder<T>::null() {
    index_.append(-1);
    hasnull_ = true;
    //return std::string("added null");
  }

  ///
  ///
  Int64TypedBuilder::Int64TypedBuilder(const ArrayBuilderOptions& options,
                                       const GrowableBuffer<int64_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string
  Int64TypedBuilder::classname() const {
    return "Int64TypedBuilder";
  };

  int64_t
  Int64TypedBuilder::length() const {
    return buffer_.length();
  }

  void
  Int64TypedBuilder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Int64TypedBuilder::snapshot() const {
    std::vector<ssize_t> shape = { (ssize_t)buffer_.length() };
    std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
    return std::make_shared<NumpyArray>(
             Identities::none(),
             util::Parameters(),
             buffer_.ptr(),
             shape,
             strides,
             0,
             sizeof(int64_t),
             util::dtype_to_format(util::dtype::int64),
             util::dtype::int64,
             kernel::lib::cpu);
  }

  bool
  Int64TypedBuilder::active() const {
    return false;
  }

  void
  Int64TypedBuilder::integer(int64_t x) {
    buffer_.append(x);
    //return std::string("added integer ") + std::to_string(x);
  }
}
