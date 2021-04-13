// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/IndexedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/IndexedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/IndexedArray.h"

namespace awkward {

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const IndexedFormPtr& form,
                                           const std::string attribute,
                                           const std::string partition)
    : form_(form),
      is_categorical_(form.get()->parameter_equals("__array__", "\"categorical\"")),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    vm_output_data_ = std::string("part")
      .append(partition_).append("-")
      .append(*form_key_).append("-")
      .append(attribute_);

    vm_func_name_ = std::string(*form_key_).append("-")
      .append(attribute_).append("-")
      .append(index_form_to_name(form_.get()->index()));

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->index()))
      .append("\n")
      .append(content_.get()->vm_output())
      .append("variable index").append("\n");

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append("dup ").append(std::to_string(static_cast<utype>(state::index)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("else").append("\n")
      .append("1 index +!").append("\n")
      .append("index @ 1- ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("then").append("\n")
      .append(";").append("\n");

    vm_error_ = content_.get()->vm_error();
  }

  const std::string
  IndexedArrayBuilder::classname() const {
    return "IndexedArrayBuilder";
  }

  const ContentPtr
  IndexedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      switch (form_.get()->index()) {
     // case Index::Form::i8:
     // case Index::Form::u8:
        case Index::Form::i32:
          return std::make_shared<IndexedArray32>(
            Identities::none(),
            form_.get()->parameters(),
            Index32(std::static_pointer_cast<int32_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len(),
                    kernel::lib::cpu),
            content_.get()->snapshot(outputs));
        case Index::Form::u32:
          return std::make_shared<IndexedArrayU32>(
            Identities::none(),
            form_.get()->parameters(),
            IndexU32(std::static_pointer_cast<uint32_t>(search->second.get()->ptr()),
                     0,
                     search->second.get()->len(),
                     kernel::lib::cpu),
            content_.get()->snapshot(outputs));
        case Index::Form::i64:
          return std::make_shared<IndexedArray64>(
            Identities::none(),
            form_.get()->parameters(),
            Index64(std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len(),
                    kernel::lib::cpu),
            content_.get()->snapshot(outputs));
        default:
          break;
      };
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs an index ")
        + FILENAME(__LINE__));
  }

  const FormPtr
  IndexedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  IndexedArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  IndexedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  IndexedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  IndexedArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  IndexedArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  IndexedArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  IndexedArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<bool>(x, data)) {
        return;
      }
    }
    content_.get()->boolean(x, builder);
  }

  void
  IndexedArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<int64_t>(x, data)) {
        return;
      }
    }
    content_.get()->int64(x, builder);
  }

  void
  IndexedArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<double>(x, data)) {
        return;
      }
    }
    content_.get()->float64(x, builder);
  }

  void
  IndexedArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<std::complex<double>>(x, data)) {
        return;
      }
    }
    content_.get()->complex(x, builder);
  }

  void
  IndexedArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'bytestring' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->bytestring(x, builder);
  }

  void
  IndexedArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'string' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->string(x, builder);
  }

  void
  IndexedArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  IndexedArrayBuilder::end_list(TypedArrayBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
