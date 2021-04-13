// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ListOffsetArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ListOffsetArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ListOffsetArray.h"

namespace awkward {

  ///
  ListOffsetArrayBuilder::ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                                                 const std::string attribute,
                                                 const std::string partition)
    : form_(form),
      is_string_builder_(form.get()->parameter_equals("__array__", "\"string\"")
        ||  form.get()->parameter_equals("__array__", "\"bytestring\"")),
      begun_(false),
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

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->offsets()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(std::to_string(static_cast<utype>(state::begin_list)))
      .append(" <> if").append("\n")
      .append(std::to_string(TypedArrayBuilder::next_error_id())).append(" err ! err @ halt").append("\n")
      .append("then").append("\n")
      .append("\n")
      .append("0").append("\n")
      .append("begin").append("\n")
      .append("pause").append("\n")
      .append("dup ")
      .append(std::to_string(static_cast<utype>(state::end_list)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" +<- stack").append("\n")
      .append("exit").append("\n")
      .append("else").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("1+").append("\n")
      .append("then").append("\n")
      .append("again").append("\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");

    vm_error_.append(content_.get()->vm_error());
    vm_error_.append("s\" ListOffsetArray Builder ")
      .append(vm_func_name_)
      .append(" needs begin_list\"").append("\n");
  }

  const std::string
  ListOffsetArrayBuilder::classname() const {
    return std::string("ListOffsetArrayBuilder ") + vm_func_name();
  }

  const ContentPtr
  ListOffsetArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      if (form_.get()->offsets() == Index::Form::i32) {
        return std::make_shared<ListOffsetArray32>(Identities::none(),
                                                   form_.get()->parameters(),
                                                   search->second.get()->toIndex32(),
                                                   content_.get()->snapshot(outputs));
      }
      else if (form_.get()->offsets() == Index::Form::u32) {
        return std::make_shared<ListOffsetArrayU32>(Identities::none(),
                                                   form_.get()->parameters(),
                                                   search->second.get()->toIndexU32(),
                                                   content_.get()->snapshot(outputs));
      }
      else if (form_.get()->offsets() == Index::Form::i64) {
        return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                   form_.get()->parameters(),
                                                   search->second.get()->toIndex64(),
                                                   content_.get()->snapshot(outputs));
      }
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs offsets")
        + FILENAME(__LINE__));
  }

  const FormPtr
  ListOffsetArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ListOffsetArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }
  const std::string
  ListOffsetArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  ListOffsetArrayBuilder::boolean(bool x, TypedArrayBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  ListOffsetArrayBuilder::int64(int64_t x, TypedArrayBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  ListOffsetArrayBuilder::float64(double x, TypedArrayBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  ListOffsetArrayBuilder::complex(std::complex<double> x, TypedArrayBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  ListOffsetArrayBuilder::bytestring(const std::string& x, TypedArrayBuilder* builder) {
    if (is_string_builder_) {
      builder->add<const std::string&>(x);
    }
    else {
      content_.get()->bytestring(x, builder);
    }
  }

  void
  ListOffsetArrayBuilder::string(const std::string& x, TypedArrayBuilder* builder) {
    if (is_string_builder_) {
      builder->add<const std::string&>(x);
    }
    else {
      content_.get()->string(x, builder);
    }
  }

  void
  ListOffsetArrayBuilder::begin_list(TypedArrayBuilder* builder) {
    if (!begun_) {
      begun_ = true;
      builder->add_begin_list();
    }
    else {
      content_.get()->begin_list(builder);
    }
  }

  void
  ListOffsetArrayBuilder::end_list(TypedArrayBuilder* builder) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (!content_.get()->active()) {
      builder->add_end_list();
      begun_ = false;
    }
    else {
      content_.get()->end_list(builder);
    }
  }

  bool
  ListOffsetArrayBuilder::active() {
    return begun_;
  }

}
