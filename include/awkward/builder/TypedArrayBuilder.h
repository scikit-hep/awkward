// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_TYPEDARRAYBUILDER_H_
#define AWKWARD_TYPEDARRAYBUILDER_H_

#include "awkward/common.h"
#include "awkward/builder/GrowableBuffer.h"

namespace awkward {
  class Content;
  using ContentPtr = std::shared_ptr<Content>;

  class Form;
  using FormPtr = std::shared_ptr<Form>;

  class FormBuilder;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  using DataPtr = std::shared_ptr<void>;

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
      snapshot() const = 0;

    virtual void
      set_input_buffer(const DataPtr& data) = 0;

    virtual void
      set_data_length(int64_t length) = 0;

    virtual bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) = 0;

  };

  // BitMaskedForm

  class LIBAWKWARD_EXPORT_SYMBOL BitMaskedArrayBuilder : public FormBuilder {
  public:

    BitMaskedArrayBuilder(const BitMaskedFormPtr& form,
                          const DataPtr& data,
                          int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const BitMaskedFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
  };

  // ByteMaskedForm

  class LIBAWKWARD_EXPORT_SYMBOL ByteMaskedArrayBuilder : public FormBuilder {
  public:

    ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form,
                           const DataPtr& data,
                           int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const ByteMaskedFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
  };

  // EmptyForm

  class LIBAWKWARD_EXPORT_SYMBOL EmptyArrayBuilder : public FormBuilder {
  public:

    EmptyArrayBuilder(const EmptyFormPtr& form,
                      const DataPtr& data,
                      int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const EmptyFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // IndexedForm

  class LIBAWKWARD_EXPORT_SYMBOL IndexedArrayBuilder : public FormBuilder {
  public:

    IndexedArrayBuilder(const IndexedFormPtr& form,
                        const DataPtr& data,
                        int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const IndexedFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
  };

  // IndexedOptionForm

  class LIBAWKWARD_EXPORT_SYMBOL IndexedOptionArrayBuilder : public FormBuilder {
  public:

    IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form,
                              const DataPtr& data,
                              int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const IndexedOptionFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
  };

  // ListForm

  class LIBAWKWARD_EXPORT_SYMBOL ListArrayBuilder : public FormBuilder {
  public:

    ListArrayBuilder(const ListFormPtr& form,
                     const DataPtr& data,
                     int64_t length,
                     bool copyarrays = true);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const ListFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
    bool copyarrays_;
  };

  // ListOffsetForm

  class LIBAWKWARD_EXPORT_SYMBOL ListOffsetArrayBuilder : public FormBuilder {
  public:

    ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                           const DataPtr& data,
                           int64_t length,
                           bool copyarrays = true);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const ListOffsetFormPtr form_;
    DataPtr data_;
    int64_t length_;
    FormBuilderPtr content_;
    bool copyarrays_;
  };

  // NumpyForm

  class LIBAWKWARD_EXPORT_SYMBOL NumpyArrayBuilder : public FormBuilder {
  public:

    NumpyArrayBuilder(const NumpyFormPtr& form,
                      const DataPtr& data,
                      int64_t length,
                      bool copyarrays = true);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const NumpyFormPtr form_;
    DataPtr data_;
    int64_t length_;
    bool copyarrays_;
  };

  // RawForm

  class LIBAWKWARD_EXPORT_SYMBOL RawArrayBuilder : public FormBuilder {
  public:

    RawArrayBuilder(const EmptyFormPtr& form,
                    const DataPtr& data,
                    int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const RawFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // RecordForm

  class LIBAWKWARD_EXPORT_SYMBOL RecordArrayBuilder : public FormBuilder {
  public:

    RecordArrayBuilder(const RecordFormPtr& form,
                       const DataPtr& data,
                       int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const RecordFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // RegularForm

  class LIBAWKWARD_EXPORT_SYMBOL RegularArrayBuilder : public FormBuilder {
  public:

    RegularArrayBuilder(const RegularFormPtr& form,
                        const DataPtr& data,
                        int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const RegularFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // UnionForm

  class LIBAWKWARD_EXPORT_SYMBOL UnionArrayBuilder : public FormBuilder {
  public:

    UnionArrayBuilder(const UnionFormPtr& form,
                      const DataPtr& data,
                      int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const UnionFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // UnmaskedForm

  class LIBAWKWARD_EXPORT_SYMBOL UnmaskedArrayBuilder : public FormBuilder {
  public:

    UnmaskedArrayBuilder(const UnmaskedFormPtr& form,
                         const DataPtr& data,
                         int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const UnmaskedFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  // VirtualForm

  class LIBAWKWARD_EXPORT_SYMBOL VirtualArrayBuilder : public FormBuilder {
  public:

    VirtualArrayBuilder(const VirtualFormPtr& form,
                        const DataPtr& data,
                        int64_t length);

    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    void
      set_input_buffer(const DataPtr& data) override {
        data_ = data;
    }

    void
      set_data_length(int64_t length) override {
        length_ = length;
    }

    bool
      apply(const FormPtr& form, const DataPtr& data, const int64_t length) override;

  private:
    const VirtualFormPtr form_;
    DataPtr data_;
    int64_t length_;
  };

  /// @class TypedArrayBuilder
  ///
  /// @brief User interface to the Builder system: the TypedArrayBuilder is a
  /// fixed reference while the Builder subclass instances change in
  /// response to accumulating data.
  class LIBAWKWARD_EXPORT_SYMBOL TypedArrayBuilder {
  public:
    /// @brief Creates an TypedArrayBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    TypedArrayBuilder(const DataPtr& data = nullptr, const int64_t length = 0);

    void
      apply(const FormPtr& form, const DataPtr& data, const int64_t length);

    /// @brief Turns the accumulated data into a Content array.
    ///
    /// This operation only converts Builder nodes into Content nodes; the
    /// buffers holding array data are shared between the Builder and the
    /// Content. Hence, taking a snapshot is a constant-time operation.
    ///
    /// It is safe to take multiple snapshots while accumulating data. The
    /// shared buffers are only appended to, which affects elements beyond
    /// the limited view of old snapshots.
    const ContentPtr
      snapshot() const;

    /// @brief Sets an Input buffer
    void
      set_input_buffer(const DataPtr& data);

    /// @brief Sets an Input buffer
    void
      set_data_length(const int64_t length);

    /// @brief Access to the internal buffer
    const DataPtr&
      data_buffer() const {
        return data_;
      }

    /// @brief Length of the internal buffer
    int64_t
      length() const {
        return length_;
      }

  private:
    /// @brief Root node of the FormBuilder tree.
    std::shared_ptr<FormBuilder> builder_;

    /// @brief Pointer to an Input buffer.
    DataPtr data_;

    /// @brief Length of a Content array.
    int64_t length_;
  };

}

#endif // AWKWARD_TYPEDARRAYBUILDER_H_
