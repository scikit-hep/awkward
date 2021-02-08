// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_TypedBuilder_H_
#define AWKWARD_TypedBuilder_H_

#include <complex>
#include <string>
#include <vector>

#include "awkward/common.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/GrowableBuffer.h"

namespace awkward {

  class Content;
  using ContentPtr = std::shared_ptr<Content>;

  /// @class TypedBuilder
  ///
  /// @brief Abstract base class for nodes within an TypedArrayBuilder
  template<typename T>
  class LIBAWKWARD_EXPORT_SYMBOL TypedBuilder {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~TypedBuilder<T>();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Current length of the accumulated array.
    virtual int64_t
      length() const = 0;

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    virtual void
      clear() = 0;

    /// @brief Turns the accumulated data into a Content array.
    ///
    /// This operation only converts TypedBuilder nodes into Content nodes; the
    /// buffers holding array data are shared between the TypedBuilder and the
    /// Content. Hence, taking a snapshot is a constant-time operation.
    ///
    /// It is safe to take multiple snapshots while accumulating data. The
    /// shared buffers are only appended to, which affects elements beyond
    /// the limited view of old snapshots.
    virtual const ContentPtr
      snapshot() const = 0;

    /// @brief If `true`, this node has started but has not finished a
    /// multi-step command (e.g. `beginX ... endX`).
    virtual bool
      active() const = 0;
  };

  /// @class BoolTypedBuilder
  ///
  /// @brief TypedBuilder node that accumulates boolean values.
  class LIBAWKWARD_EXPORT_SYMBOL BoolTypedBuilder: public TypedBuilder<uint8_t> {
    public:
      BoolTypedBuilder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<uint8_t>& buffer);

      const std::string
        classname() const override;

      int64_t
        length() const override;

      void
        clear() override;

      const ContentPtr
        snapshot() const override;

      /// @copydoc TypedBuilder::active()
      ///
      /// A BoolTypedBuilder is never active.
      bool
        active() const override;

      void
        boolean(bool x);

      private:
        const ArrayBuilderOptions options_;
        GrowableBuffer<uint8_t> buffer_;
  };

  /// @class Float64TypedBuilder
  ///
  /// @brief Builder node that accumulates real numbers (`double`).
  class LIBAWKWARD_EXPORT_SYMBOL Float64TypedBuilder:  public TypedBuilder<double> {
  public:
    Float64TypedBuilder(const ArrayBuilderOptions& options,
                        const GrowableBuffer<double>& buffer);

    const std::string
      classname() const override;

    int64_t
      length() const override;

    void
      clear() override;

    const ContentPtr
      snapshot() const override;

    /// @copydoc TypedBuilder::active()
    ///
    /// A Float64TypedBuilder is never active.
    bool
      active() const override;

    void
      real(double x);

  private:
    const ArrayBuilderOptions options_;
    GrowableBuffer<double> buffer_;
  };

  /// @class IndexedTypedBuilder
  ///
  /// @brief TypedBuilder node for accumulated data that come from an existing
  /// Content array.
  template <typename T>
  class LIBAWKWARD_EXPORT_SYMBOL IndexedTypedBuilder: public TypedBuilder<int64_t> {
  public:
    /// @brief Create an IndexedBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every TypedBuilder's constructor.
    /// @param index Contains the accumulated index (like
    /// {@link IndexedArrayOf#index IndexedArray::index}).
    /// @param array The original Content array from which the new accumulated
    /// data are drawn.
    /// @param hasnull If `true`, some of the accumulated data are missing
    /// and a #snapshot should produce an
    /// {@link IndexedArrayOf IndexedOptionArray}, rather than an
    /// {@link IndexedArrayOf IndexedArray}.
    IndexedTypedBuilder(const ArrayBuilderOptions& options,
                        const GrowableBuffer<int64_t>& index,
                        const std::shared_ptr<T>& array,
                        bool hasnull);

    /// @brief Raw pointer to the original Content `array`.
    const Content*
      arrayptr() const;

    int64_t
      length() const override;

    void
      clear() override;

    /// An IndexedBuilder is never active.
    bool
      active() const override;

    void
      null();

    /// @brief Append an element `at` a given index of an arbitrary `array`
    /// (Content instance) to the accumulated data.
    ///
    /// The resulting #snapshot will be an {@link IndexedArrayOf IndexedArray}
    /// that shares data with the provided `array`.
    virtual void
        append(const ContentPtr& array, int64_t at) = 0;

    private:
      const ArrayBuilderOptions options_;
      GrowableBuffer<int64_t> index_;
      const std::shared_ptr<T> array_;
      bool hasnull_;
  };

  class IndexedGenericBuilder: public IndexedTypedBuilder<Content> {
  public:
    IndexedGenericBuilder(const ArrayBuilderOptions& options,
                          const GrowableBuffer<int64_t>& index,
                          const ContentPtr& array,
                          bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedGenericBuilder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedI32Builder: public IndexedTypedBuilder<IndexedArray32> {
  public:
    IndexedI32Builder(const ArrayBuilderOptions& options,
                      const GrowableBuffer<int64_t>& index,
                      const std::shared_ptr<IndexedArray32>& array,
                      bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedI32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIU32Builder: public IndexedTypedBuilder<IndexedArrayU32> {
  public:
    IndexedIU32Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedArrayU32>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIU32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedI64Builder: public IndexedTypedBuilder<IndexedArray64> {
  public:
    IndexedI64Builder(const ArrayBuilderOptions& options,
                      const GrowableBuffer<int64_t>& index,
                      const std::shared_ptr<IndexedArray64>& array,
                      bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedI64Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIO32Builder: public IndexedTypedBuilder<IndexedOptionArray32> {
  public:
    IndexedIO32Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedOptionArray32>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIO32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIO64Builder: public IndexedTypedBuilder<IndexedOptionArray64> {
  public:
    IndexedIO64Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedOptionArray64>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIO64Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      append(const ContentPtr& array, int64_t at) override;
  };

  /// @class Int64TypedBuilder
  ///
  /// @brief TypedBuilder node that accumulates integers (`int64_t`).
  class LIBAWKWARD_EXPORT_SYMBOL Int64TypedBuilder: public TypedBuilder<int64_t> {
  public:
    /// @brief Create an Int64TypedBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every TypedBuilder's constructor.
    /// @param buffer Contains the accumulated integers.
    Int64TypedBuilder(const ArrayBuilderOptions& options,
                      const GrowableBuffer<int64_t>& buffer);

    const std::string
      classname() const override;

    int64_t
      length() const override;

    void
      clear() override;

    const ContentPtr
      snapshot() const override;

    /// @copydoc TypedBuilder::active()
    ///
    /// A Int64TypedBuilder is never active.
    bool
      active() const override;

    void
      integer(int64_t x);

  private:
    const ArrayBuilderOptions options_;
    GrowableBuffer<int64_t> buffer_;
  };

  // class ListTypedBuilder;
  // class OptionTypedBuilder;
  // class RecordTypedBuilder;
  // class StringTypedBuilder;
  // class TupleTypedBuilder;
  // class UnionTypedBuilder;
  // class UnknownTypedBuilder;

  template<typename T>
  void error_message(const TypedBuilder<T>& builder, const std::string& func_name) {
      std::cout << builder.classname()
        + std::string(" does not have \"")
        + func_name
        + std::string("\" method\n");
  }

  ///
  /// boolean
  ///
  template <typename T, typename = void>
  struct has_boolean
    : std::false_type {
  };

  template <typename T>
  struct has_boolean<T, decltype(std::declval<T>().boolean(bool()))>
    : std::true_type {
  };

  template <typename T>
  typename std::enable_if<has_boolean<T>::value,
                          void>::type boolean(T& builder, bool val) {
    return builder.boolean(val);
  }

  template <typename T>
  typename std::enable_if<!has_boolean<T>::value,
                          void>::type boolean(T& builder, bool val) {
    return error_message(builder, "boolean");
  }

  ///
  /// real
  ///
  template <typename T, typename = void>
  struct has_real
    : std::false_type {
  };

  template <typename T>
  struct has_real<T, decltype(std::declval<T>().real(double()))>
    : std::true_type {
  };

  template <typename T>
  typename std::enable_if<has_real<T>::value,
                          void>::type real(T& builder, double val) {
    return builder.real(val);
  }

  template <typename T>
  typename std::enable_if<!has_real<T>::value,
                          void>::type real(T& builder, double val) {
    return error_message(builder, "real");
  }

  ///
  /// complex
  ///
  template <typename T, typename = void>
  struct has_complex
    : std::false_type {
  };

  template <typename T>
  struct has_complex<T, decltype(std::declval<T>().complex(double()))>
    : std::true_type {
  };

  template <typename T>
  typename std::enable_if<has_complex<T>::value,
                          void>::type complex(T& builder, std::complex<double> val) {
    return builder.complex(val);
  }

  template <typename T>
  typename std::enable_if<!has_complex<T>::value,
                          void>::type complex(T& builder, std::complex<double> val) {
    return error_message(builder, "complex");
  }

  ///
  /// integer
  ///
  template <typename T, typename = void>
  struct has_integer
    : std::false_type {
  };

  template <typename T>
  struct has_integer<T, decltype(std::declval<T>().integer(int64_t()))>
    : std::true_type {
  };

  template <typename T>
  typename std::enable_if<has_integer<T>::value,
                          void>::type integer(T& builder, int64_t val) {
    return builder.integer(val);
  }

  template <typename T>
  typename std::enable_if<!has_integer<T>::value,
                          void>::type integer(T& builder, int64_t val) {
    return error_message(builder, "integer");
  }

  ///
  /// null
  ///
  template <typename T, typename = void>
  struct has_null
    : std::false_type {
  };

  template <typename T>
  struct has_null<T, decltype(std::declval<T>().null())>
    : std::true_type {
  };

  template <typename T>
  typename std::enable_if<has_null<T>::value,
                          void>::type null(T& builder) {
    return builder.null();
  }

  template <typename T>
  typename std::enable_if<!has_null<T>::value,
                          void>::type null(T& builder) {
    return error_message(builder, "null");
  }

  /// @class TypedArrayBuilder
  ///
  /// @brief User interface to the Builder system: the ArrayBuilder is a
  /// fixed reference while the Builder subclass instances change in
  /// response to accumulating data.
  template <typename T, typename BUILDER>
  class LIBAWKWARD_EXPORT_SYMBOL TypedArrayBuilder {
  public:
    /// @brief Creates a TypedArrayBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every TypedBuilder's constructor.
    TypedArrayBuilder<T, BUILDER>(const ArrayBuilderOptions& options)
      : builder_(std::unique_ptr<BUILDER>(new BUILDER(options, GrowableBuffer<T>::empty(options)))) {}


    /// @brief Returns a string representation of this array (single-line XML
    /// indicating the length and type).
    const std::string
      tostring() const {
        util::TypeStrs typestrs;
        typestrs["char"] = "char";
        typestrs["string"] = "string";
        std::stringstream out;
        out << "<TypedArrayBuilder length=\"" << length() << "\" type=\""
            << type(typestrs).get()->tostring() << "\"/>";
        return out.str();
      }

    /// @brief Current length of the accumulated array.
    int64_t
      length() const {
        return builder_.get()->length();
      }

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    void
      clear() {
        builder_.get()->clear();
      }

    /// @brief Current high level Type of the accumulated array.
    ///
    /// @param typestrs A mapping from `"__record__"` parameters to string
    /// representations of those types, to override the derived strings.
    const TypePtr
      type(const util::TypeStrs& typestrs) const {
        return builder_.get()->snapshot().get()->type(typestrs);
      }

    /// @brief Turns the accumulated data into a Content array.
    ///
    /// This operation only converts TypedBuilder nodes into Content nodes; the
    /// buffers holding array data are shared between the Builder and the
    /// Content. Hence, taking a snapshot is a constant-time operation.
    ///
    /// It is safe to take multiple snapshots while accumulating data. The
    /// shared buffers are only appended to, which affects elements beyond
    /// the limited view of old snapshots.
    const ContentPtr
      snapshot() const {
        return builder_.get()->snapshot();
      }

    /// @brief Adds a `null` value to the accumulated data.
    void
      null() {
        ::awkward::null(*builder_);
      }

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x) {
        ::awkward::boolean(*builder_, x);
      }

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      integer(int64_t x) {
        ::awkward::integer(*builder_, x);
      }

    /// @brief Adds a real value `x` to the accumulated data.
    void
      real(double x) {
        ::awkward::real(*builder_, x);
      }

  private:
    /// @brief Root node of the TypedBuilder tree.
    std::unique_ptr<BUILDER> builder_;
  };
}

#endif // AWKWARD_FILLABLE_H_
