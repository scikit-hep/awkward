// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/IndexedOptionArrayBuilder.cpp", line)

#include "awkward/typedbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/IndexedArray.h"

namespace awkward {

  ///
  IndexedOptionArrayBuilder::IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form,
                                                       const std::string attribute,
                                                       const std::string partition)
    : form_(form),
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

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->index()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append("dup ").append(std::to_string(static_cast<utype>(state::null)))
      .append(" = if").append("\n")
      .append("drop\n")
      .append("variable null    -1 null !").append("\n")
      .append("null @ ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("exit\n")
      .append("else\n")
      .append("variable index    1 index +!").append("\n")
      .append("index @ 1- ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("then\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");

    vm_error_ = content_.get()->vm_error();
    validate();
  }

  void
  IndexedOptionArrayBuilder::validate() const {
    if (form_.get()->parameter_equals("__array__", "\"categorical\"")) {
      throw std::invalid_argument(
        std::string("categorical form of a ") + classname()
        + std::string(" is not supported yet ")
        + FILENAME(__LINE__));
    }
  }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const ContentPtr
  IndexedOptionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      switch (form_.get()->index()) {
       // case Index::Form::i8:
          case Index::Form::i32:
            return std::make_shared<IndexedOptionArray32>(
              Identities::none(),
              form_.get()->parameters(),
              Index32(std::static_pointer_cast<int32_t>(search->second.get()->ptr()),
                      1,
                      search->second.get()->len() - 1,
                      kernel::lib::cpu),
              content_.get()->snapshot(outputs));
          case Index::Form::i64:
            return std::make_shared<IndexedOptionArray64>(
              Identities::none(),
              form_.get()->parameters(),
              Index64(std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                      1,
                      search->second.get()->len() - 1,
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
  IndexedOptionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  IndexedOptionArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  IndexedOptionArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  IndexedOptionArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  IndexedOptionArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  IndexedOptionArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    content_.get()->bytestring(x, builder);
  }

  void
  IndexedOptionArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    content_.get()->string(x, builder);
  }

  void
  IndexedOptionArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  IndexedOptionArrayBuilder::end_list(TypedArrayBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
