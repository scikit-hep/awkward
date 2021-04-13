// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/NumpyArrayBuilder.cpp", line)

#include "awkward/typedbuilder/NumpyArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/NumpyArray.h"

namespace awkward {

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const NumpyFormPtr& form,
                                       const std::string attribute,
                                       const std::string partition)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition) {
    vm_error_ = std::string("s\" NumpyForm builder accepts only ")
      .append(dtype_to_name(form_.get()->dtype())).append("\"\n");

    vm_output_data_ = std::string("part")
      .append(partition_).append("-")
      .append(*form_key_).append("-")
      .append(attribute_);

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(dtype_to_name(form_.get()->dtype())).append("\n");

    vm_func_name_ = std::string(*form_key_)
      .append("-")
      .append(dtype_to_name(form_.get()->dtype()));

    vm_func_type_ = dtype_to_state(form_.get()->dtype());

    vm_func_ = std::string(": ").append(vm_func_name()).append("\n")
      .append(vm_func_type())
      .append(" = if").append("\n")
      .append("0 data seek").append("\n")
      .append("data ").append(dtype_to_vm_format(form_.get()->dtype()))
      .append("-> ").append(vm_output_data_).append("\n")
      .append("else").append("\n")
      .append(std::to_string(TypedArrayBuilder::next_error_id())).append(" err ! err @ halt").append("\n")
      .append("then").append("\n")
      .append(";").append("\n");
  }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const ContentPtr
  NumpyArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      std::vector<ssize_t> shape = { (ssize_t)search->second.get()->len() };
      std::vector<ssize_t> strides = { (ssize_t)form_.get()->itemsize() };

      return std::make_shared<NumpyArray>(Identities::none(),
                                          form_.get()->parameters(),
                                          search->second.get()->ptr(),
                                          shape,
                                          strides,
                                          0,
                                          form_.get()->itemsize(),
                                          form_.get()->format(),
                                          util::format_to_dtype(form_.get()->format(), form_.get()->itemsize()),
                                          kernel::lib::cpu);
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs data")
        + FILENAME(__LINE__));
  }

  const FormPtr
  NumpyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  NumpyArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  NumpyArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  NumpyArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  NumpyArrayBuilder::vm_from_stack() const {
    return std::string();
  }

  const std::string
  NumpyArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  NumpyArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    builder->add<bool>(x);
  }

  void
  NumpyArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    builder->add<int64_t>(x);
  }

  void
  NumpyArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    builder->add<double>(x);
  }

  void
  NumpyArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    builder->add<std::complex<double>>(x);
  }

  void
  NumpyArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    builder->bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  NumpyArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    builder->string(x.c_str(), (int64_t)x.length());
  }

  void
  NumpyArrayBuilder::begin_list(TypedArrayBuilder* builder) {
  }

  void
  NumpyArrayBuilder::end_list(TypedArrayBuilder* builder) {
  }

}
