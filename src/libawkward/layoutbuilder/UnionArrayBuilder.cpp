// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/UnionArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/UnionArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"
#include "awkward/array/UnionArray.h"

namespace awkward {

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form,
                                       const std::string attribute,
                                       const std::string partition)
    : form_(form),
      tag_(0),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(LayoutBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition) {
    vm_func_type_ = std::to_string(static_cast<utype>(state::tag));

    for (auto const& content : form.get()->contents()) {
      contents_.push_back(LayoutBuilder::formBuilderFromA(content));
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
      vm_error_.append(contents_.back().get()->vm_error());
    }

    vm_output_tags_ = std::string("part")
      .append(partition_)
      .append("-")
      .append(*form_key_)
      .append("-tags");

    vm_output_.append("output ")
      .append(vm_output_tags_)
      .append(" ")
      .append(index_form_to_name(form_.get()->tags()))
      .append("\n");

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);
    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_).append("\n")
      .append(vm_func_type())
      .append(" = if").append("\n");

    vm_func_.append("0 data seek\n")
      .append("data ").append(index_form_to_vm_format(form_.get()->tags()))
      .append("-> stack dup ").append(vm_output_tags_).append(" <- stack\n");

    int64_t tag = 0;
    int64_t contents_size = (int64_t)contents_.size();
    bool drop = true;
    for (auto const& content : contents_) {
      drop = (tag < contents_size) ? true : false;
      if (drop) {
        vm_func_.append("dup ");
      }

      vm_func_.append(std::to_string(tag++))
        .append(" = if").append("\n");

      if (drop) {
        vm_func_.append("drop").append("\n");
        drop = false;
      }

      vm_func_.append("pause").append("\n");

      vm_func_.append(content.get()->vm_func_name())
        .append("\n")
        .append("exit").append("\n");

      vm_func_.append("then").append("\n");
    }
    vm_func_.append(std::to_string(LayoutBuilder::next_error_id()))
    .append(" err ! err @ halt").append("\n")
    .append("then\n;\n\n");

    vm_error_.append("s\" Union Array Builder error\"\n");
  }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const FormPtr
  UnionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  UnionArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  UnionArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  UnionArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  UnionArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  UnionArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  UnionArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  UnionArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  UnionArrayBuilder::tag(int8_t x) {
    tag_ = x;
  }

  void
  UnionArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->boolean(x, builder);
  }

  void
  UnionArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->int64(x, builder);
  }

  void
  UnionArrayBuilder::float64(double x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->float64(x, builder);
  }

  void
  UnionArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->complex(x, builder);
  }

  void
  UnionArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->bytestring(x, builder);
  }

  void
  UnionArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->string(x, builder);
  }

  void
  UnionArrayBuilder::begin_list(LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->begin_list(builder);
  }

  void
  UnionArrayBuilder::end_list(LayoutBuilder* builder) {
    contents_[(size_t)tag_].get()->end_list(builder);
  }

}
