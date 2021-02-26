// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_TYPEDARRAYBUILDER_H_
#define AWKWARD_TYPEDARRAYBUILDER_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthMachine.h"

#include <complex>

namespace awkward {
  class ArrayBuilderOptions;

  class Content;
  using ContentPtr = std::shared_ptr<Content>;
  class Slice;
  class Type;
  using TypePtr = std::shared_ptr<Type>;

  class Form;
  using FormPtr = std::shared_ptr<Form>;

  using ForthOtputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;

  class FormBuilder;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  using DataPtr = std::shared_ptr<uint8_t>;

  class BitMaskedForm;
  using BitMaskedFormPtr = std::shared_ptr<BitMaskedForm>;

  class ByteMaskedForm;
  using ByteMaskedFormPtr = std::shared_ptr<ByteMaskedForm>;

  class EmptyForm;
  using EmptyFormPtr = std::shared_ptr<EmptyForm>;

  class IndexedForm;
  using IndexedFormPtr = std::shared_ptr<IndexedForm>;

  class IndexedOptionForm;
  using IndexedOptionFormPtr = std::shared_ptr<IndexedOptionForm>;

  class ListForm;
  using ListFormPtr = std::shared_ptr<ListForm>;

  class ListOffsetForm;
  using ListOffsetFormPtr = std::shared_ptr<ListOffsetForm>;

  class NumpyForm;
  using NumpyFormPtr = std::shared_ptr<NumpyForm>;

  class RawForm;
  using RawFormPtr = std::shared_ptr<RawForm>;

  class RecordForm;
  using RecordFormPtr = std::shared_ptr<RecordForm>;

  class RegularForm;
  using RegularFormPtr = std::shared_ptr<RegularForm>;

  class UnionForm;
  using UnionFormPtr = std::shared_ptr<UnionForm>;

  class UnmaskedForm;
  using UnmaskedFormPtr = std::shared_ptr<UnmaskedForm>;

  class VirtualForm;
  using VirtualFormPtr = std::shared_ptr<VirtualForm>;

  struct Data {
    Data(const DataPtr& iptr, const int64_t ilength)
      : ptr(iptr),
        length(ilength) {}

    const DataPtr ptr;
    int64_t length;
  };

  struct Sum
  {
    void operator()(const std::unique_ptr<Data>& data) { length += data->length; }
    int64_t length {0};
  };

  template< class T, class U >
  std::shared_ptr<T> reinterpret_pointer_cast(const std::shared_ptr<U>& r) noexcept
  {
    auto p = reinterpret_cast<typename std::shared_ptr<T>::element_type*>(r.get());
    return std::shared_ptr<T>(r, p);
  }

  /// @class FormBuilder
  ///
  /// @brief Abstract base class for nodes within a TypedArrayBuilder
  class LIBAWKWARD_EXPORT_SYMBOL FormBuilder {
  public:
    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~FormBuilder();

    /// @brief User-friendly name of this class.
    virtual const std::string
      classname() const = 0;

    /// @brief Turns the accumulated data into a Content array.
    virtual const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const = 0;

    /// @brief
    virtual const FormPtr
      form() const = 0;

    /// @brief
    virtual void
      integer(int64_t x) {
        throw std::runtime_error(
          std::string("FIXME: 'integer' is not implemented "));
    }

    /// @brief
    virtual void
      boolean(bool x) {
        throw std::runtime_error(
          std::string("FIXME: 'boolean' is not implemented "));
    }

    /// @brief
    virtual void
      real(double x) {
        throw std::runtime_error(
          std::string("FIXME: 'real' is not implemented "));
    }

    /// @brief
    virtual const FormBuilderPtr&
      field_check(const char* key) const {
        throw std::invalid_argument(
          std::string("FIXME: 'field_check' is not implemented "));
    }

  };

  /// @class BitMaskedArrayBuilder
  ///
  /// @brief BitMaskedArray builder from a BitMaskedForm
  class LIBAWKWARD_EXPORT_SYMBOL BitMaskedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a BitMaskedArrayBuilder from a full set of parameters.
    BitMaskedArrayBuilder(const BitMaskedFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    /// @brief BitMaskedForm that defines the BitMaskedArray.
    const BitMaskedFormPtr form_;
    const FormKey form_key_;

    /// @brief Content
    FormBuilderPtr content_;
  };

  /// @class ByteMaskedArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ByteMaskedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ByteMaskedArrayBuilder from a full set of parameters.
    ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const ByteMaskedFormPtr form_;
    const FormKey form_key_;

    FormBuilderPtr content_;
  };

  /// @class EmptyArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL EmptyArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an EmptyArrayBuilder from a full set of parameters.
    EmptyArrayBuilder(const EmptyFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const EmptyFormPtr form_;
    const FormKey form_key_;
  };

  /// @class IndexedArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL IndexedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an IndexedArrayBuilder from a full set of parameters.
    IndexedArrayBuilder(const IndexedFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const IndexedFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;
  };

  /// @class IndexedOptionArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL IndexedOptionArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an IndexedOptionArrayBuilder from a full set of parameters.
    IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const IndexedOptionFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;
  };

  /// @class ListArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ListArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ListArrayBuilder from a full set of parameters.
    ListArrayBuilder(const ListFormPtr& form,
                     bool copyarrays = true);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const ListFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;
    bool copyarrays_;
  };

  /// @class ListOffsetArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ListOffsetArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ListOffsetArrayBuilder from a full set of parameters.
    ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                           bool copyarrays = true);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const ListOffsetFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;
    bool copyarrays_;
  };

  /// @class NumpyArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL NumpyArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a NumpyArrayBuilder from a full set of parameters.
    NumpyArrayBuilder(const NumpyFormPtr& form,
                      bool copyarrays = true);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    // /// @brief
    // void
    //   integer(int64_t x) override;
    //
    // /// @brief
    // void
    //   boolean(bool x) override;
    //
    // /// @brief
    // void
    //   real(double x) override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const NumpyFormPtr form_;
    const FormKey form_key_;
    bool copyarrays_;
    std::string vm_source_;
    std::string vm_output_;
  };

  /// @class RawArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL RawArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a RawArrayBuilder from a full set of parameters.
    RawArrayBuilder(const RawFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const RawFormPtr form_;
    const FormKey form_key_;
  };

  /// @class RecordArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL RecordArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a RecordArrayBuilder from a full set of parameters.
    RecordArrayBuilder(const RecordFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    // const FormBuilderPtr&
    //   field_check(const char* key) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const RecordFormPtr form_;
    const FormKey form_key_;
    std::vector<FormBuilderPtr> contents_;
    std::vector<std::string> keys_;
  };

  /// @class RegularArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL RegularArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a RegularArrayBuilder from a full set of parameters.
    RegularArrayBuilder(const RegularFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const RegularFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;
  };

  /// @class UnionArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL UnionArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a UnionArrayBuilder from a full set of parameters.
    UnionArrayBuilder(const UnionFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const UnionFormPtr form_;
    const FormKey form_key_;
    std::vector<FormBuilderPtr> contents_;
  };

  /// @class UnmaskedArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL UnmaskedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an UnmaskedArrayBuilder from a full set of parameters.
    UnmaskedArrayBuilder(const UnmaskedFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const UnmaskedFormPtr form_;
    const FormKey form_key_;
  };

  /// @class VirtualArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL VirtualArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a VirtualArrayBuilder from a full set of parameters.
    VirtualArrayBuilder(const VirtualFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

  private:
    const VirtualFormPtr form_;
    const FormKey form_key_;
  };

  /// @class UnknownFormBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL UnknownFormBuilder : public FormBuilder {
  public:
    /// @brief Creates a VirtualArrayBuilder from a full set of parameters.
    UnknownFormBuilder(const FormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

    private:
      const FormPtr form_;
  };

  /// @class TypedArrayBuilder
  ///
  /// @brief User interface to the FormBuilder system: the TypedArrayBuilder is a
  /// fixed reference while the FormBuilder subclass instances change in
  /// response to accumulating data.
  class LIBAWKWARD_EXPORT_SYMBOL TypedArrayBuilder {
  public:
    /// @brief Creates an TypedArrayBuilder from a full set of parameters.
    ///
    /// @param initial The initial number of entries for a buffer.
    TypedArrayBuilder(const FormPtr& form, const ArrayBuilderOptions& options);

    /// @brief
    void
      connect(const std::shared_ptr<ForthMachine32>& vm);

    /// @brief
    void
      debug_step() const;

    /// @brief
    const std::shared_ptr<ForthMachine32>&
      vm() const {
        return vm_;
      }

    /// @brief
    const FormPtr
      form() const;

    /// @brief
    const std::string
      to_vm() const;

    /// @brief Returns a string representation of this array (single-line XML
    /// indicating the length and type).
    const std::string
      tostring() const;

    /// @brief Current length of the accumulated array.
    int64_t
      length() const;

    /// @brief Removes all accumulated data without resetting the type
    /// knowledge.
    void
      clear();

    /// @brief Current high level Type of the accumulated array.
    ///
    /// @param typestrs A mapping from `"__record__"` parameters to string
    /// representations of those types, to override the derived strings.
    const TypePtr
      type(const util::TypeStrs& typestrs) const;

    /// @brief Turns the accumulated data into a Content array.
    ///
    /// This operation only converts FormBuilder nodes into Content nodes; the
    /// buffers holding array data are shared between the FormBuilder and the
    /// Content. Hence, taking a snapshot is a constant-time operation.
    const ContentPtr
      snapshot() const;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    const ContentPtr
      getitem_at(int64_t at) const;

    /// @brief Subinterval of this array, handling negative indexing
    /// and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// Ranges beyond the array are not an error; they are trimmed to
    /// `start = 0` on the left and `stop = length() - 1` on the right.
    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const;

    /// @brief This array with the first nested RecordArray replaced by
    /// the field at `key`.
    const ContentPtr
      getitem_field(const std::string& key) const;

    /// @brief This array with the first nested RecordArray replaced by
    /// a RecordArray of a given subset of `keys`.
    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const;

    /// @brief Entry point for general slicing: Slice represents a tuple of
    /// SliceItem nodes applying to each level of nested lists.
    const ContentPtr
      getitem(const Slice& where) const;

    /// @brief Adds a `null` value to the accumulated data.
    void
      null();

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x);

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      integer(int64_t x);

    /// @brief Adds a real value `x` to the accumulated data.
    void
      real(double x);

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x);

    /// @brief Adds an unencoded, null-terminated bytestring value `x` to the
    /// accumulated data.
    void
      bytestring(const char* x);

    /// @brief Adds an unencoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      bytestring(const char* x, int64_t length);

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x);

    /// @brief Adds a UTF-8 encoded, null-terminated bytestring value `x` to
    /// the accumulated data.
    void
      string(const char* x);

    /// @brief Adds a UTF-8 encoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      string(const char* x, int64_t length);

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x);

    /// @brief Begins building a nested list.
    ///
    /// The first 'beginlist' puts AwkwardForth VM into a state that expects
    /// another 'beginlist' or 'endlist'.
    /// The second puts the VM into a state that expects 'integer', etc.
    /// or 'endlist'.
    void
      beginlist();

    /// @brief Ends a nested list.
    void
      endlist();

    /// @brief Begins building a tuple with a fixed number of fields.
    void
      begintuple(int64_t numfields);

    /// @brief Sets the pointer to a given tuple field index; the next
    /// command will fill that slot.
    void
      index(int64_t index);

    /// @brief Ends a tuple.
    void
      endtuple();

    /// @brief Begins building a record without a name.
    ///
    /// See #beginrecord_fast and #beginrecord_check.
    void
      beginrecord();

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_fast` version of this method, a string comparison is not
    /// performed: the same pointer is assumed to have the same value each time
    /// (safe for string literals).
    ///
    /// See #beginrecord and #beginrecord_check.
    void
      beginrecord_fast(const char* name);

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `name` matches
    /// a stored `name`.
    ///
    /// See #beginrecord and #beginrecord_fast.
    void
      beginrecord_check(const char* name);

    /// @brief Begins building a record with a name.
    ///
    /// @param name This name is used to distinguish
    /// records of different types in heterogeneous data (to build a
    /// union of record arrays, rather than a record array with union
    /// fields and optional values) and it also sets the `"__record__"`
    /// parameter to later add custom behaviors in Python.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `name` matches
    /// a stored `name`.
    ///
    /// See #beginrecord and #beginrecord_fast.
    void
      beginrecord_check(const std::string& name);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_fast` version of this method, a string comparison is not
    /// performed: the same pointer is assumed to have the same value each time
    /// (safe for string literals). See #field_check.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_fast(const char* key);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `key` matches
    /// a stored `key`. See #field_fast.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_check(const char* key);

    /// @brief Sets the pointer to a given record field `key`; the next
    /// command will fill that slot.
    ///
    /// In the `_check` version of this method, a string comparison is
    /// performed every time it is called to verify that the `key` matches
    /// a stored `key`. See #field_fast.
    ///
    /// Record keys are checked in round-robin order. The best performance
    /// will be achieved by filling them in the same order for each record.
    /// Lookup time for random order scales with the number of fields.
    void
      field_check(const std::string& key);

    /// @brief Ends a record.
    void
      endrecord();

    /// @brief Append an element `at` a given index of an arbitrary `array`
    /// (Content instance) to the accumulated data, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// The resulting #snapshot will be an {@link IndexedArrayOf IndexedArray}
    /// that shares data with the provided `array`.
    void
      append(const ContentPtr& array, int64_t at);

    /// @brief Append an element `at` a given index of an arbitrary `array`
    /// (Content instance) to the accumulated data, without
    /// handling negative indexing or bounds-checking.
    ///
    /// The resulting #snapshot will be an {@link IndexedArrayOf IndexedArray}
    /// that shares data with the provided `array`.
    void
      append_nowrap(const ContentPtr& array, int64_t at);

    /// @brief Extend the accumulated data with an entire `array`.
    ///
    /// The resulting #snapshot will be an {@link IndexedArrayOf IndexedArray}
    /// that shares data with the provided `array`.
    void
      extend(const ContentPtr& array);

  private:
    /// See #initial.
    int64_t initial_;
    int64_t length_;

    std::string vm_source_;

    /// @brief Root node of the FormBuilder tree.
    std::shared_ptr<FormBuilder> builder_;

    /// @brief Current node of the FormBuilder tree.
    std::shared_ptr<FormBuilder> current_builder_;

    /// @brief
    enum class state : std::int32_t {int64 = 0, float64 = 1, begin_list = 2, end_list = 3};
    using utype = std::underlying_type<state>::type;

    std::shared_ptr<ForthMachine32> vm_;
    std::map<std::string, std::shared_ptr<ForthInputBuffer>> vm_inputs_map_;
  };

}

#endif // AWKWARD_TYPEDARRAYBUILDER_H_
