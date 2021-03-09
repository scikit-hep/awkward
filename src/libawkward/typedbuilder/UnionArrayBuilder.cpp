// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/UnionArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/UnionArray.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  UnionArrayBuilder::UnionArrayBuilder(const UnionFormPtr& form,
                                       const std::string attribute,
                                       const std::string partition)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition) {
    for (auto const& content : form.get()->contents()) {
      contents_.push_back(TypedArrayBuilder::formBuilderFromA(content));
      vm_output_.append(contents_.back().get()->vm_output());
      vm_data_from_stack_.append(contents_.back().get()->vm_from_stack());
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

    vm_output_.append("variable tag").append("\n");

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);
    for (auto const& content : contents_) {
      vm_func_.append(content.get()->vm_func());
    }
    vm_func_.append(": ")
      .append(vm_func_name_).append("\n");

    int64_t tag = 0;
    for (auto const& content : contents_) {
      vm_func_.append("dup ").append(content.get()->vm_func_type())
        .append(" = if").append("\n")
        .append(std::to_string(tag++))
        .append(" tag !").append("\n");

      vm_func_.append("tag @ ")
        .append(vm_output_tags_).append(" <- stack").append("\n");

      vm_func_.append(content.get()->vm_func_name())
        .append("\n")
        .append("exit").append("\n");

      vm_func_.append("then").append("\n");

    }

    vm_func_.append("halt\n;\n\n");

    vm_data_from_stack_.append("0 ").append(vm_output_tags_).append(" <- stack").append("\n");
  }

  const std::string
  UnionArrayBuilder::classname() const {
    return "UnionArrayBuilder";
  }

  const ContentPtr
  UnionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search_tags = outputs.find(vm_output_tags_);
    if (search_tags != outputs.end()) {
      Index8 tags(std::static_pointer_cast<int8_t>(search_tags->second.get()->ptr()),
                  1,
                  search_tags->second.get()->len() - 1,
                  kernel::lib::cpu);
      int64_t lentags = tags.length();
      Index64 current(lentags);
      Index64 outindex(lentags);
      struct Error err = kernel::UnionArray_regular_index<int8_t, int64_t>(
        kernel::lib::cpu,   // DERIVE
        outindex.data(),
        current.data(),
        lentags,
        tags.data(),
        lentags);
      util::handle_error(err, "UnionArray", nullptr);

      ContentPtrVec contents;
      for (auto content : contents_) {
        contents.push_back(content.get()->snapshot(outputs));
      }
      return UnionArray8_64(Identities::none(),
                            util::Parameters(),
                            tags,
                            outindex,
                            contents).simplify_uniontype(false, false);
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs tags and index ")
        + FILENAME(__LINE__));
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

}
